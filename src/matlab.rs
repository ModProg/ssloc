#![allow(non_snake_case, unused)]

use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::iter;
use std::ops::{Add, Bound, Index, RangeBounds};

use itertools::Itertools;
use nalgebra::{
    vector, Complex, ComplexField, DMatrix, DMatrixView, DVector, Matrix, Matrix3xX, MatrixView3xX,
};
use ndarray::{
    array, s, stack, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Axis,
    Dim,
};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

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
                    Bound::Included(max) if max == start => Some(start),
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

pub fn MBSS_locate_spec(
    x: DMatrix<F>,
    fs: F,
    micPos: Matrix3xX<F>,
    nsrc: usize,
) -> (Vec<I>, Vec<I>) {
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
    let nSamples = x.ncols();
    let nChan = x.nrows();
    assert!(nSamples > nChan, "The input signal must be in columns.");
    let nMic = micPos.ncols();
    assert_eq!(
        nMic, nChan,
        "Number of microphones and number of signal channels must be the same."
    );
    let c = 343.;
    let angularSpectrumMeth = AngularSpectrumMethod::GccPhat;
    let pooling = Pooling::Max;
    let thetaBound = -179..=180;
    let phiBound = -90..=90;
    let thetaPhiRes = 1;
    let alphaRes = 5;
    let minAngle = 1;
    let normalizeSpecInst = false;
    let thetas = thetaBound.step_by(thetaPhiRes).collect_vec();
    let phis = phiBound.step_by(thetaPhiRes).collect_vec();
    let nThetas = thetas.len();
    let nPhis = phis.len();
    // repeat thetas nPhis times
    let mut thetaGrid = Vec::with_capacity(nThetas * nPhis);
    for _ in 0..nPhis {
        thetaGrid.extend_from_slice(&thetas);
    }
    // repeat each phi nThetas times
    let mut phiGrid = Vec::with_capacity(nThetas * nPhis);
    for phi in &phis {
        phiGrid.extend(iter::repeat(phi).take(nThetas));
    }
    let specInst = MBSS_computeAngularSpectrum(
        (&x).into(),
        fs,
        angularSpectrumMeth,
        c,
        (&micPos).into(),
        &thetaGrid,
        &phiGrid,
        alphaRes,
    );
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
        Pooling::Max => specInst.map_axis(Axis(1), |a| a.iter().copied().fold(F::MIN, F::max)),
        // TODO
        // case 'sum'
        //     specGlobal = shiftdim(sum(specInst,2));
    };

    // %% Peak finding
    MBSS_findPeaks2D(
        specGlobal.view(),
        &thetas,
        &phis,
        &thetaGrid,
        &phiGrid,
        nsrc,
        minAngle,
        angularSpectrumMeth,
    )
}

/// This function search peaks in computed angular spectrum with respect to iNbSources and iMinAngle.
fn MBSS_findPeaks2D(
    ppfSpec: ArrayView1<f64>,
    piThetas: &[i64],
    piPhis: &[i64],
    piThetaGrid: &[i64],
    piPhiGrid: &[i64],
    iNbSources: usize,
    iMinAngle: i32,
    angular_spectrum_meth: AngularSpectrumMethod,
) -> (Vec<i64>, Vec<i64>) {
    let iNbThetas = piThetas.len();
    let iNbPhis = piPhis.len();

    // % Convert angular spectrum in 2D
    // (reshape(ppfSpec,iNbThetas,iNbPhis))';
    let ppfSpec2D = ppfSpec.into_shape((iNbThetas, iNbPhis)).unwrap();

    // % Estimate angular spectrum in theta and phi independently by taking
    // % the max in the corresponding direction
    let spec_theta_max = ppfSpec2D.map_axis(Axis(0), max);
    let spec_phi_max = ppfSpec2D.map_axis(Axis(1), max);

    if iNbSources == 1 {
        // % Find the maximum peak both in theta and phi direction
        let iThetaId = max_i(&spec_theta_max);
        let iPhiId = max_i(&spec_phi_max);
        (vec![piThetas[iThetaId]], vec![piPhis[iPhiId]])
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
            F::from(ppfSpec2D[(t, p)] >= max(ppfSpec2D.slice(s![t..t + 1, p..p + 1])))
        });

        // % number of local maxima
        let iNbLocalmaxima = ppiPeaks.sum() as usize;

        // % local maxima with corrresponding values
        let ppfSpec2D_peaks = (&ppfSpec2D - min(ppfSpec2D)) * ppiPeaks; // % substract min value : avoid issues (when sorting peaks) if some peaks values are negatives

        // % sort values of local maxima
        let pfSpec1D_peaks = ppfSpec2D_peaks.into_shape(iNbPhis * iNbThetas).unwrap(); // reshape(ppfSpec2D_peaks',1,iNbPhis*iNbThetas);
        let piIndexPeaks1D = sort_i_dec(pfSpec1D_peaks.view());

        let mut piEstSourcesIndex = vec![piIndexPeaks1D[0]]; //  % first source is the global maximum (first one in piSortedPeaksIndex1D)
        let mut index = 1; // search index in piSortedPeaksIndex1D
        let mut iNbSourcesFound = 1; // set to one as global maximum is already selected as source

        // %Filter the list of peaks found with respect to minAngle parameter
        while iNbSourcesFound < iNbSources && index <= iNbLocalmaxima {
            let mut bAngleAllowed = true;
            // % verify that current direction is allowed with respect to minAngle and sources already selected
            for &piEstSourceIndex in &piEstSourcesIndex {
                // % distance calculated using curvilinear abscissa (degrees) - ref. : http://geodesie.ign.fr/contenu/fichiers/Distance_longitude_latitude.pdf
                let piPhiEst = (piPhiGrid[piEstSourceIndex] as F).to_radians();
                let piPhiPeak = (piPhiGrid[piIndexPeaks1D[index]] as F).to_radians();
                let piThetaEst = (piThetaGrid[piEstSourceIndex] as F).to_radians();
                let piThetaPeak = (piThetaGrid[piIndexPeaks1D[index]] as F).to_radians();
                let dist = (piPhiEst.sin() * piPhiPeak.sin()
                    + piPhiEst.cos() * piPhiPeak.cos() * (piThetaPeak - piThetaPeak).cos())
                .acos();

                if (dist < iMinAngle as F) {
                    bAngleAllowed = false;
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

        (
            filter_index(piThetaGrid.to_vec(), &piEstSourcesIndex),
            filter_index(piPhiGrid.to_vec(), &piEstSourcesIndex),
        )
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

fn MBSS_computeAngularSpectrum(
    x: DMatrixView<F>,
    fs: F,
    angularSpectrumMeth: AngularSpectrumMethod,
    c: F,
    micPos: MatrixView3xX<F>,
    thetaGrid: &[I],
    phiGrid: &[I],
    alphaRes: usize,
) -> Array2<f64> {
    let wlen = 1024;
    let f = (1..=(wlen / 2))
        .map(|v| v as F * fs / wlen as F)
        .collect_vec();
    if angularSpectrumMeth.is_gcc() {
        // Linear transform
        let X = MBSS_stft_multi(x, wlen);
        // TODO: for some reason remove first bin
        let X = X.slice(s![1.., .., ..]);
        // X = X(2:end,:,:);
        match angularSpectrumMeth {
            AngularSpectrumMethod::GccPhat => {
                GCC_PHAT_MULTI((&X).into(), &f, c, micPos, thetaGrid, phiGrid, alphaRes)
            }
        }
    } else {
        todo!(
            r"% Quadratic transform
            hatRxx = MBSS_qstft_multi(x,fs,wlen,8,2);
            hatRxx = permute(hatRxx(:,:,2:end,:),[3 4 1 2]);"
        )
    }
}

fn MBSS_stft_multi(x: DMatrixView<F>, wlen: usize) -> Array3<C> {
    assert!(x.nrows() < x.ncols(), "the signals must be within rows.");
    assert!(wlen % 4 == 0, "the window length must be a multiple of 4.");
    // Truncate input signal to multitude of window length
    let nsampl = x.ncols() - (x.ncols() % 2 * wlen);
    let x = x.view_range(.., ..nsampl);

    // Computing STFT coefficients
    // Defining sine window
    let win = DVector::from_iterator(
        wlen,
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
            let frame = frame.view_range(.., t * wlen / 2..t * wlen / 2 + wlen);
            let mut frame = (frame.component_mul(&win.transpose()))
                .into_iter()
                .copied()
                // .map(Complex::from)
                .collect_vec();
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
    sample_points: &[usize],
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
            .try_fold(usize::MIN, |a, &e| { (a <= e).then_some(e) })
            .is_some(),
        "sample_points need to be sorted"
    );

    let interpolated = query_points
        .iter()
        .map(|&q| {
            let a = sample_points.partition_point(|&p| (p as F) <= q);
            // TODO I don't think we should need this
            if a == sample_points.len() {
                return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
            }
            if sample_points[a] as F == q {
                sample_data.column(a).to_owned()
            } else {
                // TODO I don't think we should need this
                if a + 1 == sample_points.len() {
                    return Array1::from_elem(sample_data.column(0).dim(), F::NAN);
                }
                assert!(
                    a + 1 < sample_points.len(),
                    "query_point outside of sample_points"
                );
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
    // [[nbin * nfram] * channel]
    X: ArrayView3<C>,
    f: &[F],
    c: F,
    micPos: MatrixView3xX<F>,
    thetaGrid: &[I],
    phiGrid: &[I],
    alphaRes: usize,
) -> Array2<f64> {
    let (pairId, _, alpha, alphaSampled, tauGrid) =
        MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);
    let nFrames = X.dim().1;
    let nGrid = thetaGrid.len();
    let nPairs = pairId.len();

    let mut specEntireGrid = Array3::zeros((nGrid, nFrames, nPairs));

    for i in 0..nPairs {
        let spec = phat_spec(
            (&X.select(Axis(2), &[pairId[i].0, pairId[i].1])).into(),
            f,
            &tauGrid[i],
        ); // NV % [freq x fram x local angle for each pair]
        // sum on frequencies
        // (shiftdim(sum(spec,1)))'
        let specSampledgrid = spec.sum_axis(Axis(0));
        // Order 1 interpolation on the entire grid
        // interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)')
        specEntireGrid.index_axis_mut(Axis(2), i).assign(&interp1q(
            &alphaSampled[i],
            specSampledgrid.view(),
            alpha.index_axis(Axis(0), i),
        )); // original
    }
    specEntireGrid.sum_axis(Axis(2))
}

fn MBSS_preprocess(
    c: F,
    micPos: MatrixView3xX<F>,
    thetaGrid: &[I],
    phiGrid: &[I],
    alphaRes: usize,
) -> (
    Vec<(usize, usize)>,
    Vec<f64>,
    Array2<f64>,
    Vec<Vec<usize>>,
    Vec<Vec<f64>>,
) {
    // Number of theta/phi combinations
    let nDirection = thetaGrid.len();

    // Find all microphone pair indexes
    let nMic = micPos.ncols();
    let pairId = (0..nMic)
        .flat_map(|a| (0..nMic).filter(move |&b| b != a).map(move |b| (a, b)))
        .collect_vec();
    let nMicPair = pairId.len();

    // Microphone direction vector (in xyz) for each pair
    let pfMn1n2 = pairId
        .iter()
        .map(|&(a, b)| micPos.column(a) - micPos.column(b))
        .collect_vec();

    // Microphone distance for each pair
    let dMic = pfMn1n2.iter().map(Matrix::magnitude).collect_vec();

    // Convert all potential {theta,phi} on the sphere grid in cartesian coordinates
    let Pjk = thetaGrid
        .iter()
        .zip(phiGrid.iter())
        .map(|(&theta, &phi)| {
            let theta = (theta as F).to_radians();
            let phi = (phi as F).to_radians();
            vector![phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]
        })
        .collect_vec();

    let alpha = Array2::from_shape_fn((nMicPair, nDirection), |(mic_pair, direction)| {
        let mic_pair = pfMn1n2[mic_pair];
        let direction = Pjk[direction];
        mic_pair.angle(&direction).to_degrees()
    });

    // Compute 1D angles search grids and associated TDOA (Tau) search grids for each microphone pair
    // following search grid boundaries for each microphone pair is driven by
    // the following fact : basic boundaries [0° 180°] for each pair could be
    // adapted when the global search grid does not cover the entire space
    // (leading to avoid useless curves computation and saving CPU time)
    let mut alphaSampled = Vec::with_capacity(nMicPair);
    let mut tauGrid = Vec::with_capacity(nMicPair);

    for index in 0..nMicPair {
        let (min, max) = alpha
            .row(index)
            .iter()
            .copied()
            .minmax_by(F::total_cmp)
            .into_option()
            .expect("There is at least one angle");
        let min = ((min / alphaRes as F) * alphaRes as F).floor() as usize;
        let max = ((max / alphaRes as F) * alphaRes as F).ceil() as usize;
        alphaSampled.push((min..=max).step_by(alphaRes).collect_vec());
        tauGrid.push(
            alphaSampled[index]
                .iter()
                .map(|&alpha| (alpha as f64).to_radians().cos() / c * dMic[index])
                .collect_vec(),
        );
    }
    (pairId, dMic, alpha, alphaSampled, tauGrid)
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
