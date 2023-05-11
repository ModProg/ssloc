#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_lossless,
    clippy::cast_precision_loss
)]
// TODO https://github.com/rust-ndarray/ndarray/pull/1279
#![allow(clippy::reversed_empty_ranges)]
#![cfg(feature = "wav")]
use std::path::Path;

use nalgebra::{Complex, Vector3};
use ndarray::{Array2, ArrayView2};

#[cfg(feature = "realtime")]
mod realtime;
use num::FromPrimitive;
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

    #[must_use]
    pub fn samples(&self) -> usize {
        self.data.dim().1
    }

    #[cfg(feature = "wav")]
    pub fn from_file(arg: impl AsRef<Path>) -> Self {
        use std::fs::File;
        let mut array = File::open(arg).unwrap();
        let (header, data) = wav::read(&mut array).unwrap();
        let data = data.as_sixteen().unwrap();
        Self::from_interleaved(
            data,
            header.channel_count as usize,
            header.sampling_rate as F,
        )
    }

    pub fn from_interleaved<T: ToPrimitive>(data: &[T], channels: usize, sample_rate: F) -> Self {
        Self {
            sample_rate,
            data: Array2::from_shape_fn((channels, data.len() / channels), |(c, s)| {
                data[c + s * channels].to_f64().unwrap()
            }),
        }
    }

    #[must_use]
    pub fn to_interleaved<T: FromPrimitive>(&self) -> Vec<T> {
        self.data
            .iter()
            .map(|&d| T::from_f64(d).expect("audio format can be converted"))
            .collect()
    }

    #[cfg(feature = "wav")]
    pub fn wav<T: FromPrimitive>(&self, conv: impl FnOnce(Vec<T>) -> wav::BitDepth) -> Vec<u8> {
        use std::io::Cursor;

        use wav::WAV_FORMAT_PCM;

        let mut out = Vec::new();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        wav::write(
            wav::Header::new(
                WAV_FORMAT_PCM,
                self.channels() as u16,
                self.sample_rate as u32,
                16,
            ),
            &conv(self.to_interleaved()),
            &mut Cursor::new(&mut out),
        )
        .expect("writing to buffer does not fail");
        out
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
