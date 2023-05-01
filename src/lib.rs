#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_lossless,
    clippy::cast_precision_loss
)]
// TODO https://github.com/rust-ndarray/ndarray/pull/1279
#![allow(clippy::reversed_empty_ranges)]

use nalgebra::{Complex, Vector3};
use ndarray::Array2;

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
    pub fn from_file(arg: &str) -> Self {
        use std::fs::File;
        let mut array = File::open(arg).unwrap();
        let (header, data) = wav::read(&mut array).unwrap();
        let data = data.as_sixteen().unwrap();
        Self::from_interleaved(
            data,
            header.channel_count as usize,
            header.sampling_rate as F, // , 2f64.powi(15)
        )
    }

    pub fn from_interleaved<T: ToPrimitive>(
        data: &[T],
        channels: usize,
        sample_rate: F,
        // normalize: F,
    ) -> Self {
        Self {
            sample_rate,
            data: Array2::from_shape_fn((channels, data.len() / channels), |(c, s)| {
                data[c + s * channels].to_f64().unwrap() // / normalize
            }),
        }
    }

    pub fn empty(sample_rate: F) -> Self {
        Self {
            data: Array2::default((0, 0)),
            sample_rate,
        }
    }
}
