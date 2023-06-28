#![allow(clippy::module_name_repetitions)]
use std::f64::consts::PI;
use std::fmt::{Debug, Display};
use std::iter;
use std::str::FromStr;

use itertools::Itertools;
use nalgebra::{vector, Complex, ComplexField, Matrix, Vector3};
use ndarray::{
    s, stack, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Axis, Dim,
};
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;

use crate::utils::{max, min, sort_i_dec, Step};
use crate::{Audio, Direction, Position};

type F = f64;
type C = Complex<F>;

pub struct Output {
    pub sources: Vec<(F, F)>,
    pub powers: Vec<(F, F, F)>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AngularSpectrumMethod {
    #[default]
    GccPhat,
    GccNonlin,
    Mvdr,
    Mvdrw,
    Ds,
    Dsw,
    Dnm,
    Music,
}

impl AngularSpectrumMethod {
    #[must_use]
    fn is_gcc(self) -> bool {
        matches!(self, Self::GccPhat | Self::GccNonlin)
    }
}

#[derive(Clone, Copy, Default, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    #[default]
    Max,
    Sum,
}

impl Display for Pooling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pooling::Max => write!(f, "max"),
            Pooling::Sum => write!(f, "sum"),
        }
    }
}
impl FromStr for Pooling {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "max" => Self::Max,
            "sum" => Self::Sum,
            e => return Err(format!("Unsupported pooling mode {e:?}")),
        })
    }
}

#[derive(SmartDefault, Clone, Debug, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct MbssConfig {
    pub pooling: Pooling,
    #[default = 343.0]
    pub speed_of_sound: F,
    pub spectrum_method: AngularSpectrumMethod,
    /// uses tuple, because `Range` doesn't implement Copy
    #[default((-PI, PI))]
    pub azimuth_range: (F, F),
    /// uses tuple, because `Range` doesn't implement Copy
    #[default((-PI/2.0, PI/2.0))]
    pub elevation_range: (F, F),
    #[default(1f64.to_radians())]
    pub grid_res: F,
    #[default(5f64.to_radians())]
    pub alpha_res: F,
    #[default(10f64.to_radians())]
    pub min_angle: F,
    // if(minAngle < thetaPhiRes && nscr>1), error('Minimum angle between two peaks has to be upper than theta/phi resolution'); end
    /// instantaneous local angular spectra normalization
    pub normalize_spectra: bool,
}

/// Calculate the distance between two angles using using curvilinear abscissa
///
/// ref. : <http://geodesie.ign.fr/contenu/fichiers/Distance_longitude_latitude.pdf>
#[must_use]
pub fn angular_distance(a: impl Into<Direction>, b: impl Into<Direction>) -> F {
    let a = a.into();
    let b = b.into();
    (a.elevation.sin() * b.elevation.sin()
        + a.elevation.cos() * b.elevation.cos() * (b.elevation - a.elevation).cos())
    .acos()
}

impl MbssConfig {
    pub fn create(self, mics: impl IntoIterator<Item = impl Into<Position>>) -> Mbss {
        let MbssConfig {
            pooling,
            speed_of_sound,
            spectrum_method,
            azimuth_range,
            elevation_range,
            grid_res,
            alpha_res,
            min_angle,
            normalize_spectra,
        } = self;
        let azimuth = (azimuth_range.0..azimuth_range.1)
            .step_size(grid_res)
            .collect_vec();
        let elevation = (elevation_range.0..elevation_range.1)
            .step_size(grid_res)
            .collect_vec();
        let mics = mics.into_iter().map(Into::into).collect_vec();
        let azimuth_grid = azimuth.repeat(elevation.len());
        let elevation_grid = elevation
            .iter()
            .flat_map(|&e| iter::repeat(e).take(azimuth.len()))
            .collect_vec();
        // Find all microphone pair indexes
        let pairs = (0..mics.len())
            .flat_map(|a| (0..mics.len()).filter(move |&b| b > a).map(move |b| (a, b)))
            .collect_vec();

        // Microphone direction vector (in xyz) for each pair
        let mic_dirs = pairs.iter().map(|&(a, b)| mics[a] - mics[b]).collect_vec();

        // Microphone distance for each pair
        let mic_dists = mic_dirs.iter().map(Matrix::magnitude).collect_vec();

        // Convert all potential {theta,phi} on the sphere grid in cartesian coordinates
        let angle_to_coord = azimuth_grid
            .iter()
            .zip(elevation_grid.iter())
            .map(|(&az, &el)| vector![el.cos() * az.cos(), el.cos() * az.sin(), el.sin()])
            .collect_vec();

        let alpha = Array2::from_shape_fn(
            (pairs.len(), azimuth_grid.len()),
            |(mic_pair, direction)| {
                let mic_pair = mic_dirs[mic_pair];
                let direction = angle_to_coord[direction];
                mic_pair.angle(&direction)
            },
        );

        // Compute 1D angles search grids and associated TDOA (Tau) search grids for each microphone pair
        // following search grid boundaries for each microphone pair is driven by
        // the following fact : basic boundaries [0° 180°] for each pair could be
        // adapted when the global search grid does not cover the entire space
        // (leading to avoid useless curves computation and saving CPU time)
        let mut alpha_sampled = Vec::with_capacity(pairs.len());
        let mut tau_grid = Vec::with_capacity(pairs.len());

        for index in 0..pairs.len() {
            let (min, max) = alpha
                .row(index)
                .iter()
                .copied()
                .minmax_by(F::total_cmp)
                .into_option()
                .expect("There is at least one angle");
            let min = (min / alpha_res).floor() * alpha_res;
            let max = (max / alpha_res).ceil() * alpha_res;
            alpha_sampled.push((min..=max).step_size(alpha_res).collect_vec());
            tau_grid.push(
                alpha_sampled[index]
                    .iter()
                    .map(|&alpha| alpha.cos() / speed_of_sound * mic_dists[index])
                    .collect_vec(),
            );
        }

        Mbss {
            mics,
            pooling,
            // speed_of_sound,
            spectrum_method,
            // azimuth_range,
            // elevation_range,
            // grid_res,
            // alpha_res,
            min_angle,
            normalize_spectra,
            azimuth,
            elevation,
            azimuth_grid,
            elevation_grid,
            pairs,
            alpha,
            alpha_sampled,
            tau_grid,
        }
    }
}

pub struct Mbss {
    mics: Vec<Position>,
    pooling: Pooling,
    // speed_of_sound: F,
    spectrum_method: AngularSpectrumMethod,
    // azimuth_range: Range<F>,
    // elevation_range: Range<F>,
    // grid_res: F,
    // alpha_res: F,
    min_angle: F,
    normalize_spectra: bool,
    azimuth: Vec<F>,
    elevation: Vec<F>,
    azimuth_grid: Vec<F>,
    elevation_grid: Vec<F>,
    pairs: Vec<(usize, usize)>,
    alpha: Array2<f64>,
    alpha_sampled: Vec<Vec<f64>>,
    tau_grid: Vec<Vec<f64>>,
}

pub(crate) fn wlen(sample_rate: F) -> (usize, Vec<F>) {
    let wlen = 0.064 * sample_rate as F;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let wlen = 2usize.pow(wlen.log2().ceil() as u32);
    (
        wlen,
        (1..=(wlen / 2))
            .map(|v| v as F * sample_rate / wlen as F)
            .collect_vec(),
    )
}

impl Mbss {
    #[must_use]
    pub fn array_centroid(&self) -> Position {
        self.mics
            .iter()
            .fold(Vector3::zeros(), |a, v| a + v / self.mics.len() as F)
    }

    #[must_use]
    pub fn analyze_spectrum(&self, audio: &Audio) -> Array2<F> {
        assert_eq!(
            self.mics.len(),
            audio.channels(),
            "Number of microphones and number of signal channels must be the same.",
        );

        let (wlen, f) = wlen(audio.sample_rate());

        let x_ft;
        let x_ft = if self.spectrum_method.is_gcc() {
            // Linear transform
            x_ft = stft_multi(audio.data.view(), wlen);
            // TODO: for some reason remove first bin
            x_ft.slice(s![1usize.., .., ..])
            // X = X(2:end,:,:);
            // match angularSpectrumMeth {
            //     AngularSpectrumMethod::GccPhat => {
            //         GCC_PHAT_MULTI((&X).into(), &f, c, micPos, thetaGrid, phiGrid, alphaRes)
            //     }
            // }
        } else {
            todo!(
                r"% Quadratic transform
            hatRxx = MBSS_qstft_multi(x,fs,wlen,8,2);
            hatRxx = permute(hatRxx(:,:,2:end,:),[3 4 1 2]);"
            )
        };

        // let (pairs, /* _distances, */ alpha, alpha_sampled, tau_grid) = preprocess(
        //     self.speed_of_sound,
        //     &self.mics,
        //     &self.azimuth_grid,
        //     &self.elevation_grid,
        //     self.alpha_res,
        // );

        let spec_inst = match self.spectrum_method {
            AngularSpectrumMethod::GccPhat => self.gcc_phat_multi(x_ft.view(), &f),
            other => todo!("{other:?}"),
        };
        // %% Normalize instantaneous local angular spectra if requested
        if self.normalize_spectra {
            todo!(
                r"[~,nFrames,~] = size(specInst);
    for i=1:nFrames
        minVal = min(min(specInst(:,i)));
    specInst(:,i)=(specInst(:,i) - minVal)/ max(max(specInst(:,i)- minVal));
    end
        "
            );
        }

        // %% Pooling
        let spec = match self.pooling {
            Pooling::Max => spec_inst.map_axis(Axis(1), |a| a.iter().copied().fold(F::MIN, F::max)),
            Pooling::Sum => spec_inst.sum_axis(Axis(1)),
        };

        // State {
        //     spec,
        //     azimuth: self.azimuth,
        //     elevation: self.elevation,
        //     azimuth_grid: self.azimuth_grid,
        //     elevation_grid: self.elevation_grid,
        //     nsrc,
        //     min_angle: self.min_angle,
        // }
        spec.into_shape((self.n_elevations(), self.n_azimuth()))
            .unwrap()
    }

    #[must_use]
    pub fn spectrum(&self, spec: ArrayView2<F>, threshold: F) -> Vec<(Direction, F)> {
        let mut out = Vec::new();
        assert_eq!(spec.nrows(), self.elevation.len());
        assert_eq!(spec.ncols(), self.azimuth.len());
        for (row, &el) in spec.rows().into_iter().zip(self.elevation.iter()) {
            for (&value, &az) in row.into_iter().zip(self.azimuth.iter()) {
                if value > threshold {
                    out.push((Direction::new(az, el), value));
                }
            }
        }
        out
    }

    #[must_use]
    pub fn find_sources(&self, spec: ArrayView2<F>, nsrc: usize) -> Vec<(Direction, f64)> {
        self.find_peaks(nsrc, spec)
    }

    #[must_use]
    pub fn locate_spec(&self, audio: &Audio, nsrc: usize) -> Vec<(Direction, F)> {
        self.find_sources(self.analyze_spectrum(audio).view(), nsrc)
    }

    fn n_elevations(&self) -> usize {
        self.elevation.len()
    }

    fn n_azimuth(&self) -> usize {
        self.azimuth.len()
    }

    /// This function search peaks in computed angular spectrum
    fn find_peaks(&self, nsrc: usize, spec: ArrayView2<f64>) -> Vec<(Direction, F)> {
        //    % search all local maxima (local maximum : value higher than all neighborhood values)
        //    % some alternative implementations using matlab image processing toolbox are explained here :
        //    % http://stackoverflow.com/questions/22218037/how-to-find-local-maxima-in-image)
        //
        // % Current implementation uses no specific toolbox. Explanations can be found with following link :
        //    % http://stackoverflow.com/questions/5042594/comparing-matrix-element-with-its-neighbours-without-using-loop-in-matlab
        //    % All values of flat peaks are detected as peaks with this implementation :
        //  ones(size(ppfSpec2D,1)+2,size(ppfSpec2D,2)+2) * -Inf;
        let mut ppf_padpeak_filter =
            Array2::from_elem(spec.raw_dim() + Dim((2, 2)), F::NEG_INFINITY);
        ppf_padpeak_filter
            .slice_mut(s![1isize..-1, 1isize..-1])
            .assign(&spec);

        let ((el, az), m) = spec
            .indexed_iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        println!(
            "{:?}",
            (
                self.azimuth[az].to_degrees(),
                self.elevation[el].to_degrees(),
                m
            )
        );

        // % Find peaks : compare values with their neighbours
        // ppiPeaks = ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,2:end-1) & ... % top
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  2:end-1) & ... % bottom
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,1:end-2) & ... % right
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,3:end)   & ... % left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,1:end-2) & ... % top/left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,3:end)   & ... % top/right
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  1:end-2) & ... % bottom/left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  3:end);        % bottom/right
        let ppi_peaks = Array2::from_shape_fn(spec.dim(), |(t, p)| {
            u32::from(
                spec[(t, p)]
                    >= max(spec.slice(s![
                        t.saturating_sub(1)..(t + 2).min(spec.dim().0),
                        p.saturating_sub(1)..(p + 2).min(spec.dim().1)
                    ])),
            ) as F
        });

        // % number of local maxima
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let nlocal_maximas = ppi_peaks.sum() as usize;

        // % local maxima with corrresponding values
        let ppf_spec2_d_peaks = (&spec - min(spec)) * ppi_peaks; // % substract min value : avoid issues (when sorting peaks) if some peaks values are negatives

        // % sort values of local maxima
        let pf_spec1_d_peaks = ppf_spec2_d_peaks
            .into_shape(self.n_azimuth() * self.n_elevations())
            .unwrap(); // reshape(ppfSpec2D_peaks',1,iNbPhis*iNbThetas);
        let pi_index_peaks1_d = sort_i_dec(pf_spec1_d_peaks.view());

        let mut pi_est_sources_index = vec![pi_index_peaks1_d[0]]; //  % first source is the global maximum (first one in piSortedPeaksIndex1D)
        let mut index = 1; // search index in piSortedPeaksIndex1D
        let mut nb_sources_found = 1; // set to one as global maximum is already selected as source

        // %Filter the list of peaks found with respect to minAngle parameter
        while nb_sources_found < nsrc && index <= nlocal_maximas {
            let mut angle_allowed = true;
            // % verify that current direction is allowed with respect to minAngle and sources already selected
            for &est_source in &pi_est_sources_index {
                // % distance calculated using curvilinear abscissa (degrees) - ref. : http://geodesie.ign.fr/contenu/fichiers/Distance_longitude_latitude.pdf
                let el_est = self.elevation_grid[est_source];
                let el_peak = self.elevation_grid[pi_index_peaks1_d[index]];
                let az_est = self.azimuth_grid[est_source];
                let az_peak = self.azimuth_grid[pi_index_peaks1_d[index]];
                let dist = angular_distance((az_est, el_est), (az_peak, el_peak));

                if dist < self.min_angle as F {
                    angle_allowed = false;
                    break;
                }
            }

            // % store new source
            if angle_allowed {
                pi_est_sources_index.push(pi_index_peaks1_d[index]);
                nb_sources_found += 1;
            }

            index += 1;
        }

        pi_est_sources_index
            .into_iter()
            .map(|i| {
                (
                    Direction::new(self.azimuth_grid[i], self.elevation_grid[i]),
                    pf_spec1_d_peaks[i],
                )
            })
            .collect()
        // (
        //     filter_index(azimuth_grid.to_vec(), &piEstSourcesIndex),
        //     filter_index(elevation_grid.to_vec(), &piEstSourcesIndex),
        // )
    }

    // }

    fn gcc_phat_multi(
        &self,
        // pairs: &[(usize, usize)],
        // // distances: &[F],
        // alpha: ArrayView2<F>,
        // alpha_sampled: &[Vec<F>],
        // // azimuth: &[F],
        // // elevation: &[F],
        // tau_grid: &[Vec<F>],
        // // c: F,
        x_ft: ArrayView3<C>,
        f: &[F],
    ) -> Array2<f64> {
        let mut spec_inst = Array2::zeros((self.alpha.ncols(), x_ft.dim().1));

        for (i, pair) in self.pairs.iter().enumerate() {
            let spec = phat_spec(
                (&x_ft.select(Axis(2), &[pair.0, pair.1])).into(),
                f,
                &self.tau_grid[i],
            ); // NV % [freq x fram x local angle for each pair]
            // sum on frequencies
            // (shiftdim(sum(spec,1)))'

            let spec_sampledgrid = spec.sum_axis(Axis(0));
            let interp = interp1q(
                &self.alpha_sampled[i],
                spec_sampledgrid.view(),
                self.alpha.index_axis(Axis(0), i),
            );
            // Order 1 interpolation on the entire grid
            // interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)')
            spec_inst += &interp; // original
        }
        spec_inst
    }
}

/// Returns bins x frames x channels
pub(crate) fn stft_multi(x: ArrayView2<F>, wlen: usize) -> Array3<C> {
    assert!(x.nrows() < x.ncols(), "the signals must be within rows.");
    assert!(wlen % 4 == 0, "the window length must be a multiple of 4.");
    // Truncate input signal to multitude of window length
    let nsampl = x.ncols() - (x.ncols() % 2 * wlen);
    let x = x.slice(s![.., ..nsampl]);

    // Computing STFT coefficients
    // Defining sine window
    let win = Array1::from_iter(
        // wlen,
        (0..wlen).map(|i| ((i as F + 0.5) / wlen as F * PI).sin()),
    );
    let nfram = nsampl / wlen * 2 - 1;
    let nbin = wlen / 2 + 1;

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(wlen);
    // let mut planner = FftPlanner::new();
    // let fft = planner.plan_fft_forward(wlen);

    let mut x_ft = Array3::default((nbin, nfram, x.nrows()));

    for channel in 0..x.nrows() {
        let mut x_ft = x_ft.index_axis_mut(ndarray::Axis(2), channel);
        for t in 0..nfram {
            // Framing
            let frame = x.row(channel);
            let frame = frame.slice(s![t * wlen / 2..t * wlen / 2 + wlen]);
            let mut frame = (&frame * &win).to_vec();
            // let mut frame = (0..wlen).map(|i| ((i as F /50.).sin()).into()).collect_vec();
            // FFT
            let mut frame_ft = vec![Complex::default(); nbin];

            fft.process(&mut frame, &mut frame_ft).unwrap();
            // let frame = &frame[0..nbin];
            x_ft.index_axis_mut(ndarray::Axis(1), t)
                .assign(&Array1::from(frame_ft));
        }
    }
    x_ft
}

fn interp1q(
    sample_points: &[F],
    sample_data: ArrayView2<F>,
    query_points: ArrayView1<F>,
) -> Array2<F> {
    assert_eq!(
        sample_points.len(),
        sample_data.ncols(),
        "sample_points and sample_data need to have the same length"
    );
    debug_assert!(
        sample_points
            .iter()
            .try_fold(F::MIN, |a, &e| { (a <= e).then_some(e) })
            .is_some(),
        "sample_points need to be sorted"
    );

    let interpolated = query_points
        .iter()
        .map(|&q| {
            // eprintln!("{q:?} in {sample_points:?}");
            let a = sample_points.partition_point(|&p| (p as F) <= q);
            // TODO I don't think we should need this
            if a == sample_points.len() {
                assert!((sample_points.last().unwrap() - q).abs() < 0.01);
                return sample_data.column(a - 1).to_owned();
            }
            // if a == 0 {
            //     return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
            // }
            assert!(a > 0);
            assert!(a < sample_points.len());
            let a = a - 1;
            #[allow(clippy::float_cmp)]
            if sample_points[a] == q {
                sample_data.column(a).to_owned()
            } else {
                // TODO I really don't think we should need this
                // if a + 1 == sample_points.len() {
                //     return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
                // }
                let b = a + 1;
                let lerp =
                    (q - sample_points[a] as F) / (sample_points[b] as F - sample_points[a] as F);
                let a = &sample_data.column(a);
                let b = &sample_data.column(b);

                a + lerp * (b - a)
            }
        })
        .collect_vec();
    let smth = stack(
        Axis(0),
        interpolated
            .iter()
            .map(ArrayBase::view)
            .collect_vec()
            .as_slice(),
    )
    .unwrap();
    smth
}

// PHAT_SPEC Computes the GCC-PHAT spectrum as defined in
// C. Knapp, G. Carter, "The generalized cross-correlation method for
// estimation of time delay", IEEE Transactions on Acoustics, Speech and
// Signal Processing, 24(4):320â327, 1976.
//
// spec = phat_spec(X, f, tauGrid)
//
// Inputs:
// X: nbin x nFrames x 2 matrix containing the STFT coefficients of the input
//     signal in all time-frequency bins
// f: nbin x 1 vector containing the center frequency of each frequency bin
//     in Hz
// tauGrid: 1 x ngrid vector of possible TDOAs in seconds
//
// Output:
// spec: nbin x nFrames x ngrid array of angular spectrum values
//
// This calculates
// [i] = (X1 * X2.conj()) / | (X1 * X2.conj()) e^( tau * -2 * i * pi  * f)
fn phat_spec(x_ft: ArrayView3<C>, f: &[F], tau_grid: &[F]) -> Array3<f64> {
    let x_ft1 = x_ft.index_axis(Axis(2), 0);
    let x_ft2 = x_ft.index_axis(Axis(2), 1);

    let (nbin, n_frames) = x_ft1.dim();
    let ngrid = tau_grid.len();

    let mut spec = Array3::zeros((nbin, n_frames, ngrid));
    let mut p = &x_ft1.to_owned() * &x_ft2.map(C::conj);
    p /= &p.map(|c| c.modulus().into());

    for (pk_ind, tau) in tau_grid.iter().enumerate().take(ngrid) {
        // exp(-2*1i*pi*tauGrid(pkInd)*f);
        let exp: Array1<_> = f
            .iter()
            .map(|f| (-2. * C::i() * PI * tau * f).exp())
            .collect_vec()
            .into();
        // EXP = EXP(:,temp);
        let exp = stack(Axis(1), &vec![exp.view(); n_frames]).unwrap();
        // spec(:,:,pkInd) = real(P.*EXP);
        spec.index_axis_mut(Axis(2), pk_ind)
            .assign(&(&p * exp).mapv(Complex::real));
    }
    spec
}
