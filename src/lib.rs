#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_lossless,
    clippy::cast_precision_loss
)]
// TODO https://github.com/rust-ndarray/ndarray/pull/1279
#![allow(clippy::reversed_empty_ranges)]
//! Sound source localization crate.
//!
//! Currently very much unstable and undocumented, but that should be fixed in
//! the comming weeks.
#[cfg(feature = "image")]
use std::fmt::Write;
#[cfg(feature = "wav")]
use std::path::Path;

use nalgebra::{Complex, UnitQuaternion, Vector3};
#[cfg(feature = "image")]
use ndarray::ArrayView2;
use ndarray::{Array2, Axis};
use num::FromPrimitive;

#[cfg(feature = "realtime")]
mod realtime;
#[cfg(feature = "realtime")]
pub use realtime::*;
mod utils;

pub type F = f64;
pub type I = i64;
pub type C = Complex<F>;
pub type Position = Vector3<F>;

pub use mbss::{Mbss, MbssConfig};
use realfft::num_traits::ToPrimitive;
pub mod mbss;
mod sss;
pub use sss::delay_and_sum;

#[must_use]
pub struct Audio {
    data: Array2<F>,
    sample_rate: F,
}

impl Audio {
    #[must_use]
    pub fn channels(&self) -> usize {
        self.data.dim().0
    }

    pub fn retain_channels(&mut self, mut filter: impl FnMut(usize) -> bool) {
        for channel in (0..self.channels()).rev() {
            if !filter(channel) {
                self.data.remove_index(Axis(0), channel);
            }
        }
    }

    #[must_use]
    pub fn samples(&self) -> usize {
        self.data.dim().1
    }

    #[must_use]
    pub fn sample_rate(&self) -> F {
        self.sample_rate
    }

    #[cfg(feature = "wav")]
    pub fn from_file(arg: impl AsRef<Path>) -> Self {
        use std::fs::File;
        Self::from_wav(File::open(arg).unwrap())
    }

    #[cfg(feature = "wav")]
    pub fn from_wav<R: std::io::Read>(data: R) -> Self {
        use itertools::Itertools;

        let reader = hound::WavReader::new(data).unwrap();
        let spec = reader.spec();
        match spec.sample_format {
            hound::SampleFormat::Float => {
                let data = reader.into_samples();
                Self::from_interleaved(
                    &data.collect::<Result<Vec<f32>, _>>().unwrap(),
                    spec.channels as usize,
                    spec.sample_rate as F,
                )
            }
            hound::SampleFormat::Int => {
                // https://web.archive.org/web/20230605122301/https://gist.github.com/endolith/e8597a58bcd11a6462f33fa8eb75c43d
                let normalize: Box<dyn Fn(i32) -> F> = match spec.bits_per_sample {
                    u @ ..=8 => Box::new(move |s: i32| {
                        (s - 2i32.pow(u as u32 - 1)) as F / (2f64.powi(u as i32 - 1) - 1.)
                    }),
                    i => Box::new(move |s: i32| s as F / (2f64.powi(i as i32) - 1.)),
                };
                let data = reader.into_samples();
                Self::from_interleaved(
                    &data
                        .map_ok(normalize)
                        .collect::<Result<Vec<F>, _>>()
                        .unwrap(),
                    spec.channels as usize,
                    spec.sample_rate as F,
                )
            }
        }
    }

    pub fn from_interleaved<T: ToPrimitive>(data: &[T], channels: usize, sample_rate: F) -> Self {
        Self {
            sample_rate,
            data: Array2::from_shape_fn((channels, data.len() / channels), |(c, s)| {
                data[c + s * channels].to_f64().unwrap()
            }),
        }
    }

    pub fn to_interleaved<T: FromPrimitive>(&self) -> impl Iterator<Item = T> + '_ {
        self.data
            .iter()
            .map(|&d| T::from_f64(d).expect("audio format can be converted"))
    }

    #[cfg(feature = "wav")]
    #[must_use]
    pub fn wav(&self, sample_format: hound::SampleFormat, bits_per_sample: u16) -> Vec<u8> {
        use std::io::Cursor;

        use itertools::Itertools;

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let spec = hound::WavSpec {
            channels: self.channels().try_into().unwrap(),
            sample_rate: self.sample_rate as u32,
            bits_per_sample,
            sample_format,
        };

        let mut out = Vec::new();
        let mut writer = hound::WavWriter::new(Cursor::new(&mut out), spec).unwrap();

        match sample_format {
            hound::SampleFormat::Float => self
                .to_interleaved::<f32>()
                .map(|sample| writer.write_sample(sample))
                .try_collect()
                .unwrap(),
            hound::SampleFormat::Int => todo!(),
        }
        writer.finalize().unwrap();
        out
    }

    fn get(&self, mic: usize, sample: usize) -> Option<F> {
        self.data.get((mic, sample)).copied()
    }
}

#[cfg(feature = "image")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn spec_to_image(spectrum: ArrayView2<F>) -> image::GrayImage {
    use image::{GrayImage, Luma};
    let normalize = spectrum
        .iter()
        .copied()
        .max_by(F::total_cmp)
        .expect("spectrum is not empty")
        / u8::MAX as F;
    let mut img = GrayImage::new(spectrum.ncols() as u32, spectrum.nrows() as u32);
    for ((y, x), value) in spectrum.indexed_iter() {
        img.put_pixel(
            x as u32,
            (spectrum.nrows() - 1 - y) as u32,
            Luma([(value / normalize) as u8]),
        );
    }
    img
}

#[cfg(feature = "image")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[must_use]
pub fn spec_to_csv(spectrum: ArrayView2<F>) -> String {
    let mut out = String::new();
    for row in spectrum.rows() {
        for col in row.iter() {
            write!(out, "{col},").expect("string writing does not fail");
        }
        writeln!(out).expect("string writing does not fail");
    }
    out
}

/// Converts angles azimuth and elevation to the respective position on a
/// unitsphere around the microphone array.
#[must_use]
pub fn angles_to_unit_vec(az: F, el: F) -> Position {
    angles_to_quaternion(az, el).transform_vector(&Position::new(1., 0., 0.))
}

/// Converts angles azimuth and elevation to the matching quaternion
#[must_use]
pub fn angles_to_quaternion(az: F, el: F) -> UnitQuaternion<F> {
    UnitQuaternion::from_euler_angles(0., -el, az)
}
