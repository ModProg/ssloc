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
use derive_more::Constructor;
use nalgebra::{Complex, UnitQuaternion, Vector3};
#[cfg(feature = "image")]
use ndarray::ArrayView2;

#[cfg(feature = "realtime")]
mod realtime;
#[cfg(feature = "realtime")]
pub use realtime::*;
pub mod mbss;
pub use mbss::{Mbss, MbssConfig};
mod sss;
pub use sss::DelayAndSum;
mod utils;

mod audio;
pub use audio::*;

pub type F = f64;
pub type I = i64;
pub type C = Complex<F>;
pub type Position = Vector3<F>;

#[derive(Debug, Clone, Copy, PartialEq, Default, Constructor)]
pub struct Direction {
    pub azimuth: F,
    pub elevation: F,
}

impl From<(F, F)> for Direction {
    fn from((azimuth, elevation): (F, F)) -> Self {
        Self::new(azimuth, elevation)
    }
}

impl Direction {
    /// Converts angles azimuth and elevation to the respective position on a
    /// unitsphere around the microphone array.
    #[must_use]
    pub fn to_unit_vec(self) -> Position {
        self.to_quaternion()
            .transform_vector(&Position::new(1., 0., 0.))
    }

    /// Converts angles azimuth and elevation to the matching quaternion
    #[must_use]
    pub fn to_quaternion(self) -> UnitQuaternion<F> {
        UnitQuaternion::from_euler_angles(0., -self.elevation, self.azimuth)
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
