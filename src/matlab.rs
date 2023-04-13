#![allow(non_snake_case, unused)]

use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::fmt::{Debug, Display, Write};
use std::ops::{Add, Bound, Index, RangeBounds};
use std::{fs, iter};

use itertools::Itertools;
use nalgebra::{
    vector, Complex, ComplexField, DMatrix, DMatrixView, DVector, Matrix, Matrix3xX, MatrixView3xX,
    Vector3,
};
use ndarray::{
    array, s, stack, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Axis,
    Dim,
};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::{Audio, Position};

type F = f64;
type I = i64;
type C = Complex<F>;

#[derive(Clone, Copy)]
enum AngularSpectrumMethod {
    GccPhat,
}

#[derive(Clone, Copy)]
enum Pooling {
    Max,
    // TODO Sum
}

trait Step<T> {
    type Iter: Iterator<Item = T>;
    fn step_size(self, step_size: T) -> Self::Iter;
}

impl<R: RangeBounds<F>> Step<F> for R {
    type Iter = Box<dyn Iterator<Item = F>>;

    fn step_size(self, step_size: F) -> Self::Iter {
        let mut start = match self.start_bound() {
            Bound::Included(f) => *f,
            Bound::Excluded(f) => f + step_size,
            Bound::Unbounded => F::MIN,
        };
        assert!(!start.is_nan(), "not a number");
        let end = self.end_bound().clone().cloned();
        Box::new(iter::once(start).chain(iter::from_fn(move || {
            if start == f64::INFINITY && matches!(end, Bound::Unbounded) {
                Some(start)
            } else {
                let next = start + step_size;
                assert_ne!(start, next, "step size too small, did not increase float");
                start = next;
                match end {
                    // TODO check if this is a reasonable implementation of approx_eq
                    Bound::Included(max) if (max - start).abs() < 0.00001 => Some(start),
                    Bound::Included(max) | Bound::Excluded(max) if max > start => Some(start),
                    _ => None,
                }
            }
        })))
    }
}

impl AngularSpectrumMethod {
    #[must_use]
    fn is_gcc(&self) -> bool {
        matches!(self, Self::GccPhat)
    }
}

fn min<'a>(iter: impl IntoIterator<Item = &'a F>) -> F {
    iter.into_iter().copied().reduce(F::min).unwrap()
}
fn max<'a>(iter: impl IntoIterator<Item = &'a F>) -> F {
    iter.into_iter().copied().reduce(F::max).unwrap()
}
fn max_i<'a>(iter: impl IntoIterator<Item = &'a F>) -> usize {
    let mut max = F::MIN;
    let mut max_i = 0;
    for (idx, &value) in iter.into_iter().enumerate() {
        if value > max {
            max_i = idx;
            max = value;
        }
    }
    assert_ne!(max, F::MIN, "Expect to find a non F::MIN element");
    max_i
}

fn sort_i_dec<
    I: Index<usize, Output = F> + IntoIterator<IntoIter = II> + Copy,
    II: ExactSizeIterator,
>(
    list: I,
) -> Vec<usize> {
    let mut indices = (0..list.into_iter().len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| list[b].total_cmp(&list[a]));
    indices
}

fn csv_dump(array: ArrayView2<impl Display>, filename: &str) {
    let mut s = String::new();
    for row in array.rows() {
        for col in row.into_iter() {
            write!(s, "{col},").unwrap();
        }
        writeln!(s);
    }
    fs::write(filename.to_string() + ".csv", s).unwrap();
}

fn csv_dump_2d(
    array: impl IntoIterator<Item = impl IntoIterator<Item = impl Debug>>,
    filename: &str,
) {
    let mut s = String::new();
    for row in array {
        for col in row {
            write!(s, "{col:?},").unwrap();
        }
        writeln!(s);
    }
    fs::write(format!("{filename}.csv"), s).unwrap();
}

fn csv_dump_1d(array: impl IntoIterator<Item = impl Debug>, filename: &str) {
    let mut s = String::new();
    for row in array {
        writeln!(s, "{row:?}");
    }
    fs::write(format!("{filename}.csv"), s).unwrap();
}

pub fn MBSS_locate_spec(audio: Audio, mic_pos: &[Position], nsrc: usize) -> Vec<(F, F)> {
    // TODO
    // if nargin<4, error('Not enough input arguments.'); end
    // [nSamples,nChan]=size(x);
    // [nMic,~] = size(micPos);
    // fprintf('Input signal duration: %.02f seconds\n',nSamples/fs);
    // if nChan>nSamples, error('The input signal must be in columns.'); end
    // if nChan~=nMic, error('Number of microphones and number of signal channels must be the same'); end
    // if nargin < 5, c = 343; end
    // if nargin < 6, angularSpectrumMeth = 'GCC-PHAT'; end
    // if ~any(strcmp(angularSpectrumMeth, {'GCC-PHAT' 'GCC-NONLIN' 'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'})), error('Unknown local angular spectrum.'); end
    // if nargin < 7, pooling = 'max'; end
    // if ~any(strcmp(pooling, {'max' 'sum'})), error('Unknown pooling function.'); end
    // if nargin < 8, thetaBound = [-179 180]; end
    // if (length(thetaBound) ~= 2), error('Length of thetaBound must be 2'); end
    // if (thetaBound(1) >= thetaBound(2)), error('thetaBound must be filled in ascending order'); end
    // if nargin < 9, phiBound = [-90 90]; end
    // if (length(phiBound) ~= 2), error('Length of phiBound must be 2'); end
    // if (phiBound(1) >= phiBound(2)), error('phiBound must be filled in ascending order'); end
    // if nargin < 10, thetaPhiRes = 1; end
    // if nargin < 11, alphaRes = 5; end
    // if nargin < 12, minAngle = 1; end
    // if(minAngle < thetaPhiRes && nscr>1), error('Minimum angle between two peaks has to be upper than theta/phi resolution'); end
    // if nargin < 13, normalizeSpecInst = 0; end
    // if nargin < 14, specDisplay = 0; end
    //
    // if((nsrc == 1) && normalizeSpecInst)
    //     warning('Use of instantaneous local angular spectra normalization with one source to be located is unnecessary. Switch to no normalization usage.');
    //     normalizeSpecInst = 0;
    // end
    assert_eq!(
        mic_pos.len(),
        audio.channels(),
        "Number of microphones and number of signal channels must be the same.",
    );
    let c = 343.;
    let angular_spectrum_meth = AngularSpectrumMethod::GccPhat;
    let pooling = Pooling::Max;
    let az_bound = (-179f64).to_radians()..=180f64.to_radians();
    let elevation_bound = (-90f64).to_radians()..=90f64.to_radians();
    let grid_res = 1f64.to_radians();
    let alpha_res = 5f64.to_radians();
    let min_angle = 10f64.to_radians();
    let normalizeSpecInst = false;
    let azimuth = az_bound.step_size(grid_res).collect_vec();
    let elevation = elevation_bound.step_size(grid_res).collect_vec();
    // repeat thetas nPhis times
    let azimuth_grid = azimuth.repeat(elevation.len());
    // repeat each phi nThetas times
    let elevation_grid = elevation
        .iter()
        .flat_map(|&e| iter::repeat(e).take(azimuth.len()))
        .collect_vec();
    let array_centroid = mic_pos
        .iter()
        .fold(Vector3::zeros(), |a, v| a + v / mic_pos.len() as F);

    let wlen = 0.064 * audio.sample_rate as F;
    let wlen = 2usize.pow(wlen.log2().ceil() as u32);

    let X;
    let X = if angular_spectrum_meth.is_gcc() {
        // Linear transform
        X = MBSS_stft_multi(audio.data.view(), wlen);
        // TODO: for some reason remove first bin
        X.slice(s![1usize.., .., ..])
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

    let (pairs, distances, alpha, alpha_sampled, tau_grid) =
        MBSS_preprocess(c, mic_pos, &azimuth_grid, &elevation_grid, alpha_res);
    let f = (1..=(wlen / 2))
        .map(|v| v as F * audio.sample_rate / wlen as F)
        .collect_vec();

    let spec_inst = match angular_spectrum_meth {
        AngularSpectrumMethod::GccPhat => GCC_PHAT_MULTI(
            &pairs,
            &distances,
            alpha.view(),
            &alpha_sampled,
            &azimuth,
            &elevation,
            &tau_grid,
            c,
            X.view(),
            &f,
        ),
    };
    // %% Normalize instantaneous local angular spectra if requested
    if normalizeSpecInst {
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
    let specGlobal = match pooling {
        Pooling::Max => spec_inst.map_axis(Axis(1), |a| a.iter().copied().fold(F::MIN, F::max)),
        // TODO
        // case 'sum'
        //     specGlobal = shiftdim(sum(specInst,2));
    };

    // %% Peak finding
    MBSS_findPeaks2D(
        specGlobal.view(),
        &azimuth,
        &elevation,
        &azimuth_grid,
        &elevation_grid,
        nsrc,
        min_angle,
        angular_spectrum_meth,
    )
}

/// This function search peaks in computed angular spectrum with respect to iNbSources and iMinAngle.
fn MBSS_findPeaks2D(
    ppfSpec: ArrayView1<F>,
    azimuth: &[F],
    elevation: &[F],
    azimuth_grid: &[F],
    elevation_grid: &[F],
    nsrc: usize,
    min_angle: F,
    angular_spectrum_meth: AngularSpectrumMethod,
) -> Vec<(F, F)> {
    // % Convert angular spectrum in 2D
    // (reshape(ppfSpec,iNbThetas,iNbPhis))';
    let ppfSpec2D = ppfSpec
        .into_shape((elevation.len(), azimuth.len()))
        .unwrap();

    // % Estimate angular spectrum in theta and phi independently by taking
    // % the max in the corresponding direction
    let spec_theta_max = ppfSpec2D.map_axis(Axis(0), max);
    let spec_phi_max = ppfSpec2D.map_axis(Axis(1), max);

    if nsrc == 1 {
        // TODO verify this does what it should
        // % Find the maximum peak both in theta and phi direction
        let iThetaId = max_i(&spec_theta_max);
        let iPhiId = max_i(&spec_phi_max);
        vec![(azimuth[iThetaId], elevation[iPhiId])]
    } else {
        //    % search all local maxima (local maximum : value higher than all neighborhood values)
        //    % some alternative implementations using matlab image processing toolbox are explained here :
        //    % http://stackoverflow.com/questions/22218037/how-to-find-local-maxima-in-image)
        //
        // % Current implementation uses no specific toolbox. Explanations can be found with following link :
        //    % http://stackoverflow.com/questions/5042594/comparing-matrix-element-with-its-neighbours-without-using-loop-in-matlab
        //    % All values of flat peaks are detected as peaks with this implementation :
        //  ones(size(ppfSpec2D,1)+2,size(ppfSpec2D,2)+2) * -Inf;
        let mut ppfPadpeakFilter =
            Array2::from_elem(ppfSpec2D.raw_dim() + Dim((2, 2)), F::NEG_INFINITY);
        ppfPadpeakFilter
            .slice_mut(s![1isize..-1, 1isize..-1])
            .assign(&ppfSpec2D);

        // % Find peaks : compare values with their neighbours
        // ppiPeaks = ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,2:end-1) & ... % top
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  2:end-1) & ... % bottom
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,1:end-2) & ... % right
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,3:end)   & ... % left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,1:end-2) & ... % top/left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,3:end)   & ... % top/right
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  1:end-2) & ... % bottom/left
        //            ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  3:end);        % bottom/right
        let ppiPeaks = Array2::from_shape_fn(ppfSpec2D.dim(), |(t, p)| {
            F::from(
                ppfSpec2D[(t, p)]
                    >= max(ppfSpec2D.slice(s![
                        t.saturating_sub(1)..(t + 2).min(ppfSpec2D.dim().0),
                        p.saturating_sub(1)..(p + 2).min(ppfSpec2D.dim().1)
                    ])),
            )
        });

        // % number of local maxima
        let iNbLocalmaxima = ppiPeaks.sum() as usize;

        // % local maxima with corrresponding values
        let ppfSpec2D_peaks = (&ppfSpec2D - min(ppfSpec2D)) * ppiPeaks; // % substract min value : avoid issues (when sorting peaks) if some peaks values are negatives

        // % sort values of local maxima
        let pfSpec1D_peaks = ppfSpec2D_peaks
            .into_shape(azimuth.len() * elevation.len())
            .unwrap(); // reshape(ppfSpec2D_peaks',1,iNbPhis*iNbThetas);
        let piIndexPeaks1D = sort_i_dec(pfSpec1D_peaks.view());

        let mut piEstSourcesIndex = vec![piIndexPeaks1D[0]]; //  % first source is the global maximum (first one in piSortedPeaksIndex1D)
        let mut index = 1; // search index in piSortedPeaksIndex1D
        let mut iNbSourcesFound = 1; // set to one as global maximum is already selected as source

        // %Filter the list of peaks found with respect to minAngle parameter
        while iNbSourcesFound < nsrc && index <= iNbLocalmaxima {
            let mut bAngleAllowed = true;
            // % verify that current direction is allowed with respect to minAngle and sources already selected
            for &piEstSourceIndex in &piEstSourcesIndex {
                // % distance calculated using curvilinear abscissa (degrees) - ref. : http://geodesie.ign.fr/contenu/fichiers/Distance_longitude_latitude.pdf
                let piPhiEst = (elevation_grid[piEstSourceIndex]);
                let piPhiPeak = (elevation_grid[piIndexPeaks1D[index]]);
                let piThetaEst = (azimuth_grid[piEstSourceIndex]);
                let piThetaPeak = (azimuth_grid[piIndexPeaks1D[index]]);
                let dist = (piPhiEst.sin() * piPhiPeak.sin()
                    + piPhiEst.cos() * piPhiPeak.cos() * (piThetaPeak - piThetaPeak).cos())
                .acos();

                if (dist < min_angle as F) {
                    bAngleAllowed = false;
                    panic!("I DONT WANT TO BREAK so soon {index}");
                    break;
                }
            }

            // % store new source
            if bAngleAllowed {
                piEstSourcesIndex.push(piIndexPeaks1D[index]);
                iNbSourcesFound += 1;
            }

            index = index + 1;
        }

        piEstSourcesIndex
            .into_iter()
            .map(|i| (azimuth_grid[i], elevation_grid[i]))
            .collect()
        // (
        //     filter_index(azimuth_grid.to_vec(), &piEstSourcesIndex),
        //     filter_index(elevation_grid.to_vec(), &piEstSourcesIndex),
        // )
    }
}

fn filter_index<'a, T>(
    list: impl IntoIterator<Item = T>,
    indexes: impl IntoIterator<Item = &'a usize>,
) -> Vec<T> {
    let indexes: BTreeSet<_> = indexes.into_iter().collect();
    list.into_iter()
        .enumerate()
        .filter_map(|(i, e)| indexes.contains(&i).then_some(e))
        .collect()
}

// fn MBSS_computeAngularSpectrum(
//     angularSpectrumMeth: AngularSpectrumMethod,
//     pairs: &[(usize, usize)],
//     distances: &[F],
//     alpha: ArrayView2<F>,
//     alpha_sampled: &[Vec<F>],
//     azimuth: &[F],
//     elevation: &[F],
//     tau_grid: &[Vec<F>],
//     c: F,
//     X: ArrayView3<C>,
//     f: &[F],
// ) -> Array2<f64> {
//     let wlen = 1024;
//     if angularSpectrumMeth.is_gcc() {
//         // Linear transform
//         let X = MBSS_stft_multi(audio, wlen);
//         // TODO: for some reason remove first bin
//         let X = X.slice(s![1.., .., ..]);
//         // X = X(2:end,:,:);
//         match angularSpectrumMeth {
//             AngularSpectrumMethod::GccPhat => {
//                 GCC_PHAT_MULTI((&X).into(), &f, c, micPos, thetaGrid, phiGrid, alphaRes)
//             }
//         }
//     } else {
//         todo!(
//             r"% Quadratic transform
//             hatRxx = MBSS_qstft_multi(x,fs,wlen,8,2);
//             hatRxx = permute(hatRxx(:,:,2:end,:),[3 4 1 2]);"
//         )
//     }
// }
//
fn MBSS_stft_multi(x: ArrayView2<F>, wlen: usize) -> Array3<C> {
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

    let mut X = Array3::default((nbin, nfram, x.nrows()));

    for channel in 0..x.nrows() {
        let mut X = X.index_axis_mut(ndarray::Axis(2), channel);
        for t in 0..nfram {
            // Framing
            let frame = x.row(channel);
            let frame = frame.slice(s![t * wlen / 2..t * wlen / 2 + wlen]);
            let mut frame = (&frame * &win).to_vec();
            // let mut frame = (0..wlen).map(|i| ((i as F /50.).sin()).into()).collect_vec();
            // FFT
            let mut fframe = vec![Complex::default(); nbin];

            fft.process(&mut frame, &mut fframe);
            // let frame = &frame[0..nbin];
            X.index_axis_mut(ndarray::Axis(1), t)
                .assign(&Array1::from(fframe));
        }
    }
    X
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
            let a = sample_points.partition_point(|&p| (p as F) <= q);
            // TODO I don't think we should need this
            // if a == sample_points.len() {
            //     return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
            // }
            // if a == 0 {
            //     return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
            // }
            assert!(a > 0);
            assert!(a < sample_points.len());
            let a = a - 1;
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

fn GCC_PHAT_MULTI(
    pairs: &[(usize, usize)],
    distances: &[F],
    alpha: ArrayView2<F>,
    alpha_sampled: &[Vec<F>],
    azimuth: &[F],
    elevation: &[F],
    tau_grid: &[Vec<F>],
    c: F,
    X: ArrayView3<C>,
    f: &[F],
) -> Array2<f64> {
    let mut spec_inst = Array2::zeros((alpha.ncols(), X.dim().1));

    for (i, pair) in pairs.iter().enumerate() {
        let spec = phat_spec(
            (&X.select(Axis(2), &[pair.0, pair.1])).into(),
            f,
            &tau_grid[i],
        ); // NV % [freq x fram x local angle for each pair]
           // sum on frequencies
           // (shiftdim(sum(spec,1)))'

        let specSampledgrid = spec.sum_axis(Axis(0));
        let interp = interp1q(
            &alpha_sampled[i],
            specSampledgrid.view(),
            alpha.index_axis(Axis(0), i),
        );
        // Order 1 interpolation on the entire grid
        // interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)')
        spec_inst += &interp; // original
    }
    spec_inst
}
//
fn MBSS_preprocess(
    c: F,
    mic_pos: &[Position],
    az_grid: &[F],
    el_grid: &[F],
    alpha_res: F,
) -> (
    Vec<(usize, usize)>,
    Vec<F>,
    Array2<F>,
    Vec<Vec<F>>,
    Vec<Vec<F>>,
) {
    assert_eq!(az_grid.len(), el_grid.len());

    // Find all microphone pair indexes
    let pair_ids = (0..mic_pos.len())
        .flat_map(|a| {
            (0..mic_pos.len())
                .filter(move |&b| b > a)
                .map(move |b| (a, b))
        })
        .collect_vec();

    // Microphone direction vector (in xyz) for each pair
    let mic_dirs = pair_ids
        .iter()
        .map(|&(a, b)| mic_pos[a] - mic_pos[b])
        .collect_vec();

    // Microphone distance for each pair
    let mic_dists = mic_dirs.iter().map(Matrix::magnitude).collect_vec();

    // Convert all potential {theta,phi} on the sphere grid in cartesian coordinates
    let angle_to_coord = az_grid
        .iter()
        .zip(el_grid.iter())
        .map(|(&az, &el)| vector![el.cos() * az.cos(), el.cos() * az.sin(), el.sin()])
        .collect_vec();

    let alpha = Array2::from_shape_fn((pair_ids.len(), az_grid.len()), |(mic_pair, direction)| {
        let mic_pair = mic_dirs[mic_pair];
        let direction = angle_to_coord[direction];
        mic_pair.angle(&direction)
    });

    // Compute 1D angles search grids and associated TDOA (Tau) search grids for each microphone pair
    // following search grid boundaries for each microphone pair is driven by
    // the following fact : basic boundaries [0° 180°] for each pair could be
    // adapted when the global search grid does not cover the entire space
    // (leading to avoid useless curves computation and saving CPU time)
    let mut alpha_sampled = Vec::with_capacity(pair_ids.len());
    let mut tau_grid = Vec::with_capacity(pair_ids.len());

    for index in 0..pair_ids.len() {
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
                .map(|&alpha| (alpha as f64).cos() / c * mic_dists[index])
                .collect_vec(),
        );
    }
    (pair_ids, mic_dists, alpha, alpha_sampled, tau_grid)
}

// // PHAT_SPEC Computes the GCC-PHAT spectrum as defined in
// // C. Knapp, G. Carter, "The generalized cross-correlation method for
// // estimation of time delay", IEEE Transactions on Acoustics, Speech and
// // Signal Processing, 24(4):320â327, 1976.
// //
// // spec = phat_spec(X, f, tauGrid)
// //
// // Inputs:
// // X: nbin x nFrames x 2 matrix containing the STFT coefficients of the input
// //     signal in all time-frequency bins
// // f: nbin x 1 vector containing the center frequency of each frequency bin
// //     in Hz
// // tauGrid: 1 x ngrid vector of possible TDOAs in seconds
// //
// // Output:
// // spec: nbin x nFrames x ngrid array of angular spectrum values
// //
fn phat_spec(X: ArrayView3<C>, f: &[F], tauGrid: &[F]) -> Array3<f64> {
    let X1 = X.index_axis(Axis(2), 0);
    let X2 = X.index_axis(Axis(2), 1);

    let (nbin, nFrames) = X1.dim();
    let ngrid = tauGrid.len();

    let mut spec = Array3::zeros((nbin, nFrames, ngrid));
    let mut P = (&X1.to_owned() * &X2.map(C::conj));
    P /= &P.map(|c| c.modulus().into());

    for pkInd in 0..ngrid {
        // exp(-2*1i*pi*tauGrid(pkInd)*f);
        let EXP: Array1<_> = f
            .into_iter()
            .map(|f| (-2. * C::i() * PI * tauGrid[pkInd] * f).exp())
            .collect_vec()
            .into();
        // EXP = EXP(:,temp);
        let EXP = stack(Axis(1), &vec![EXP.view(); nFrames]).unwrap();
        // spec(:,:,pkInd) = real(P.*EXP);
        spec.index_axis_mut(Axis(2), pkInd)
            .assign(&(&P * EXP).mapv(Complex::real));
    }
    spec
}
