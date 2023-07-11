use std::f64::consts::PI;
use std::ops::Add;

use itertools::Itertools;
use nalgebra::ComplexField;
use smart_default::SmartDefault;

use crate::{Audio, Direction, Position, F};

/// Uses a Delay-and-Sum Beamformer to extract a single channel.
#[derive(SmartDefault, Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct DelayAndSum {
    #[default = 343.0]
    pub speed_of_sound: F,
    /// Number of samples used for interpolating between samples audio data.
    #[default(Some(11))]
    pub filter: Option<usize>,
    pub mics: Vec<Position>,
}

impl DelayAndSum {
    /// Returns the expected audio length produced by delay and sum, if filter is enabled, it
    /// will remove a few frames at the start and end of the audio.
    #[must_use]
    pub fn expected_len(&self, audio: &Audio) -> usize {
        audio.samples() - self.filter.map(|f| f + f % 2).unwrap_or_default()
    }

    /// Uses a Delay-and-Sum Beamformer to extract a single channel in the direction specified via `az` and `el`.
    pub fn beam_form<'a>(
        &'a self,
        direction: impl Into<Direction>,
        audio: &'a Audio,
    ) -> impl Iterator<Item = F> + 'a {
        let target = direction.into().to_unit_vec();

        let mics = self
            .mics
            .iter()
            .map(|m| (m - target).magnitude() / self.speed_of_sound)
            .collect_vec();

        let min = mics
            .iter()
            .copied()
            .min_by(F::total_cmp)
            .expect("at least one mic");

        let mics = mics.into_iter().map(|m| m - min).collect_vec();

        assert_eq!(audio.channels(), mics.len());

        // https://web.archive.org/web/20230605140150/http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
        let sub_sample: Box<dyn Fn(usize, F) -> Option<F>> = match self.filter {
            Some(filter_length) => Box::new(move |mic: usize, delay: F| {
                let filter_length = filter_length + filter_length % 2;
                let centre_tap = filter_length / 2;
                (0..filter_length)
                    .map(move |t| {
                        let x = t as F - (delay - delay.floor());
                        let sinc = (PI * (x - centre_tap as F)).sinc();
                        let window = 0.54 - 0.46 * (2. * PI * (x + 0.5) / filter_length as F).cos();
                        let tap_weight = window * sinc;

                        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                        {
                            Some(audio.get(mic, t + delay as usize - centre_tap)? * tap_weight)
                        }
                    })
                    .fold_options(0., F::add)
            }),
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            None => Box::new(|mic: usize, delay: F| audio.get(mic, delay.round() as usize)),
        };

        // we skip the first centre_tap samples, hopefully fine
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        (self.filter.unwrap_or_default() / 2 + 1..).map_while(move |i| {
            mics.iter()
                .enumerate()
                .map(|(mic, tau)| {
                    sub_sample(mic, i as F + *tau * audio.sample_rate)
                        .map(|s| s / audio.channels() as F)
                })
                .fold_options(0., F::add)
        })
    }
}

// /// Uses a MVDR Beamformer to extract a single channel.
// #[derive(SmartDefault, Clone, Debug, PartialEq)]
// pub struct Mvdr {
//     #[default = 343.0]
//     pub speed_of_sound: F,
//     pub mics: Vec<Position>,
// }
//
// impl Mvdr {
//     /// Uses a MVDR Beamformer to extract a single channel in the direction specified via `az` and `el`.
//     pub fn beam_form<'a>(
//         &'a self,
//         direction: impl Into<Direction>,
//         audio: &'a Audio,
//     ) -> impl Iterator<Item = F> + 'a {
//         let target: Position = direction.into().to_unit_vec();
//         let window_len = (0.64 * audio.sample_rate) as usize;
//
//         // Do mvdr on each stft frame
//
//         (0..audio.samples() / window_len).flat_map(move |i| {
//             let data = audio.data.slice(s![
//                 ..,
//                 i * window_len..((i + 1) * window_len).min(audio.samples())
//             ]);
//             let mut csm: DMatrix<C> = data
//                 .axis_iter(Audio::SAMPLE)
//                 .map(|mics| {
//                     let mics = Vector::from(mics.mapv(C::from_real).to_vec());
//                     let mics_adjoint = mics.adjoint();
//                     mics * mics_adjoint
//                 })
//                 .sum();
//             assert!(csm.try_inverse_mut(), "should be able to invert csm");
//
//             // v = e^-j 2 pi f delta_m
//             let v = DVector::from_iterator(
//                 self.mics.len(),
//                 self.mics.iter().map(|m| {
//                     (-C::i()
//                             * 100. // f
//                             * 2.
//                             * PI
//                             * ((m - target).magnitude() / self.speed_of_sound))
//                         .exp()
//                 }),
//             );
//             // w_H = (v^H S_xx^-1)/(v^H S_xx^-1 v)
//             let lhs = v.adjoint() * &csm;
//             let rhs = v.adjoint() * &csm * v;
//             let w_h = lhs / rhs.x;
//
//             data.axis_iter(Audio::SAMPLE)
//                 .map(move |audio| {
//                     (w_h.clone() * Vector::from(audio.mapv(C::from_real).to_vec()))
//                         .x
//                         .real()
//                 })
//                 .collect_vec()
//         })
//
//         // let target = direction.into().to_unit_vec();
//         //
//         // let mics = self
//         //     .mics
//         //     .iter()
//         //     .map(|m| (m - target).magnitude() / self.speed_of_sound)
//         //     .collect_vec();
//         //
//         // let min = mics
//         //     .iter()
//         //     .copied()
//         //     .min_by(F::total_cmp)
//         //     .expect("at least one mic");
//         //
//         // let mics = mics.into_iter().map(|m| m - min).collect_vec();
//         //
//         // assert_eq!(audio.channels(), mics.len());
//         //
//         // // https://web.archive.org/web/20230605140150/http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
//         // let sub_sample: Box<dyn Fn(usize, F) -> Option<F>> = match self.filter {
//         //     Some(filter_length) => Box::new(move |mic: usize, delay: F| {
//         //         let filter_length = filter_length + filter_length % 2;
//         //         let centre_tap = filter_length / 2;
//         //         (0..filter_length)
//         //             .map(move |t| {
//         //                 let x = t as F - (delay - delay.floor());
//         //                 let sinc = (PI * (x - centre_tap as F)).sinc();
//         //                 let window = 0.54 - 0.46 * (2. * PI * (x + 0.5) / filter_length as F).cos();
//         //                 let tap_weight = window * sinc;
//         //
//         //                 #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
//         //                 {
//         //                     Some(audio.get(mic, t + delay as usize - centre_tap)? * tap_weight)
//         //                 }
//         //             })
//         //             .fold_options(0., F::add)
//         //     }),
//         //     #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
//         //     None => Box::new(|mic: usize, delay: F| audio.get(mic, delay.round() as usize)),
//         // };
//         //
//         // // we skip the first centre_tap samples, hopefully fine
//         // #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
//         // (self.filter.unwrap_or_default() / 2 + 1..).map_while(move |i| {
//         //     mics.iter()
//         //         .enumerate()
//         //         .map(|(mic, tau)| {
//         //             sub_sample(mic, i as F + *tau * audio.sample_rate)
//         //                 .map(|s| s / audio.channels() as F)
//         //         })
//         //         .fold_options(0., F::add)
//         // })
//     }
// }
//
// // impl Mvdr {
// //     /// Uses a MVDR Beamformer to extract a single channel in the direction specified via `az` and `el`.
// //     pub fn beam_form<'a>(&'a self, direction: impl Into<Direction>, audio: &'a Audio)
// //     // -> impl Iterator<Item = F> + 'a
// //     {
// //         let (wlen, f) = wlen(audio.sample_rate);
// //         let target: Position = direction.into().to_unit_vec();
// //
// //         // Do mvdr on each stft frame
// //
// //         let win = Array1::from_iter(
// //             // wlen,
// //             (0..wlen).map(|i| ((i as F + 0.5) / wlen as F * PI).sin()),
// //         );
// //         let nfram = audio.samples() / wlen * 2 - 1;
// //         let nbin = wlen / 2 + 1;
// //
// //         let mut planner = RealFftPlanner::<F>::new();
// //         let fft = planner.plan_fft_forward(wlen);
// //         let fftb = planner.plan_fft_inverse(wlen);
// //
// //         let signal: Array1<F> = Default::default();
// //             let csm: DMatrix<C> = audio
// //                 .data
// //                 .slice(s![.., frame])
// //                 .axis_iter(Audio::SAMPLE)
// //                 .map(|mics| {
// //                     let mics = Vector::from(mics.mapv(C::from_real).to_vec());
// //                     let mics_adjoint = mics.adjoint();
// //                     mics * mics_adjoint
// //                 })
// //                 .sum();
// //             assert!(csm.try_inverse_mut(), "should be able to invert csm");
// //
// //         for t in 0..nfram {
// //             let frame = t * wlen / 2..t * wlen / 2 + wlen;
// //             // S_xx = E[x * x^H]
// //             let csm: DMatrix<C> = audio
// //                 .data
// //                 .slice(s![.., frame])
// //                 .axis_iter(Audio::SAMPLE)
// //                 .map(|mics| {
// //                     let mics = Vector::from(mics.mapv(C::from_real).to_vec());
// //                     let mics_adjoint = mics.adjoint();
// //                     mics * mics_adjoint
// //                 })
// //                 .sum();
// //             assert!(csm.try_inverse_mut(), "should be able to invert csm");
// //
// //             // v = e^-j 2 pi f delta_m
// //             let v = |bin: usize| {
// //                 DVector::from_iterator(
// //                     self.mics.len(),
// //                     self.mics.iter().map(|m| {
// //                         (-C::i()
// //                             * f[bin]
// //                             * 2.
// //                             * PI
// //                             * ((m - target).magnitude() / self.speed_of_sound))
// //                             .exp()
// //                     }),
// //                 )
// //             };
// //             // w_H = (v^H S_xx^-1)/(v^H S_xx^-1 v)
// //             let w_h = |bin| {
// //                 let v = v(bin);
// //                 let lhs = v.adjoint() * csm;
// //                 let rhs = v.adjoint() * csm * v;
// //                 lhs / rhs.x
// //             };
// //             for channel in 0..audio.channels() {
// //                 let mut x_ft = x_ft.index_axis_mut(ndarray::Axis(2), channel);
// //                 // Framing
// //                 let frame = x.row(channel);
// //                 let frame = frame.slice(s![t * wlen / 2..t * wlen / 2 + wlen]);
// //                 let mut frame = (&frame * &win).to_vec();
// //                 // let mut frame = (0..wlen).map(|i| ((i as F /50.).sin()).into()).collect_vec();
// //                 // FFT
// //                 let mut frame_ft = vec![Complex::default(); nbin];
// //
// //                 fft.process(&mut frame, &mut frame_ft).unwrap();
// //                 // let frame = &frame[0..nbin];
// //                 x_ft.index_axis_mut(ndarray::Axis(1), t)
// //                     .assign(&Array1::from(frame_ft));
// //             }
// //         }
// //         // S_xx = E[XX^H]
// //         let mut cross_spectral_matrix =
// //             Array3::default((audio.channels(), audio.channels(), f.len()));
// //
// //         let fft = crate::mbss::stft_multi(audio.data.view(), wlen);
// //         // S_xx = E[x x^H], x is real
// //
// //         for frame in fft.axis_iter(Axis(1)) {
// //             for (i, bin) in frame.axis_iter(Axis(0)).enumerate() {
// //                 let bin = Vector::from(bin.to_vec());
// //                 let bin_adjoint = bin.adjoint();
// //                 let res = bin * bin_adjoint;
// //                 let res =
// //                     Array2::from_shape_fn((audio.channels(), audio.channels()), |idx| res[idx])
// //                         .mapv(Complex::real);
// //                 cross_spectral_matrix
// //                     .index_axis_mut(Axis(2), i)
// //                     .add_assign(&res);
// //             }
// //         }
// //
// //         cross_spectral_matrix /= fft.len_of(Axis(1)) as F;
// //
// //         let target: Position = direction.into().to_unit_vec();
// //
// //         let v = |bin: usize, m: usize| {
// //             (-C::i()
// //                 * f[bin]
// //                 * 2.
// //                 * PI
// //                 * ((self.mics[m] - target).magnitude() / self.speed_of_sound))
// //                 .exp()
// //         };
// //         // let target = direction.into().to_unit_vec();
// //         //
// //         // let mics = self
// //         //     .mics
// //         //     .iter()
// //         //     .map(|m| (m - target).magnitude() / self.speed_of_sound)
// //         //     .collect_vec();
// //         //
// //         // let min = mics
// //         //     .iter()
// //         //     .copied()
// //         //     .min_by(F::total_cmp)
// //         //     .expect("at least one mic");
// //         //
// //         // let mics = mics.into_iter().map(|m| m - min).collect_vec();
// //         //
// //         // assert_eq!(audio.channels(), mics.len());
// //         //
// //         // // https://web.archive.org/web/20230605140150/http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
// //         // let sub_sample: Box<dyn Fn(usize, F) -> Option<F>> = match self.filter {
// //         //     Some(filter_length) => Box::new(move |mic: usize, delay: F| {
// //         //         let filter_length = filter_length + filter_length % 2;
// //         //         let centre_tap = filter_length / 2;
// //         //         (0..filter_length)
// //         //             .map(move |t| {
// //         //                 let x = t as F - (delay - delay.floor());
// //         //                 let sinc = (PI * (x - centre_tap as F)).sinc();
// //         //                 let window = 0.54 - 0.46 * (2. * PI * (x + 0.5) / filter_length as F).cos();
// //         //                 let tap_weight = window * sinc;
// //         //
// //         //                 #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
// //         //                 {
// //         //                     Some(audio.get(mic, t + delay as usize - centre_tap)? * tap_weight)
// //         //                 }
// //         //             })
// //         //             .fold_options(0., F::add)
// //         //     }),
// //         //     #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
// //         //     None => Box::new(|mic: usize, delay: F| audio.get(mic, delay.round() as usize)),
// //         // };
// //         //
// //         // // we skip the first centre_tap samples, hopefully fine
// //         // #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
// //         // (self.filter.unwrap_or_default() / 2 + 1..).map_while(move |i| {
// //         //     mics.iter()
// //         //         .enumerate()
// //         //         .map(|(mic, tau)| {
// //         //             sub_sample(mic, i as F + *tau * audio.sample_rate)
// //         //                 .map(|s| s / audio.channels() as F)
// //         //         })
// //         //         .fold_options(0., F::add)
// //         // })
// //     }
// // }
