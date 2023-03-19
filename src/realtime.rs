use std::fmt::Debug;
use std::str::FromStr;

use alsa::pcm::{self, Access, HwParams, IoFormat};
use alsa::{Direction, ValueOr, PCM};
use derive_more::Display;
use num::{Num, ToPrimitive};
use serde::{Deserialize, Serialize};

use crate::Audio;

#[derive(Clone, Copy, Display, Deserialize, Serialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    F32,
    F64,
}
impl FromStr for Format {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "s8" => Format::S8,
            "u8" => Format::U8,
            "s16" => Format::S16,
            "u16" => Format::U16,
            "s32" => Format::S32,
            "u32" => Format::U32,
            "f32" => Format::F32,
            "f64" => Format::F64,
            e => return Err(format!("Unsupported audio format {e:?}")),
        })
    }
}

impl From<Format> for pcm::Format {
    fn from(value: Format) -> Self {
        match value {
            Format::S8 => Self::S8,
            Format::U8 => Self::U8,
            Format::S16 => Self::s16(),
            Format::U16 => Self::u16(),
            Format::S32 => Self::s32(),
            Format::U32 => Self::u32(),
            Format::F32 => Self::float(),
            Format::F64 => Self::float64(),
        }
    }
}

impl PartialEq<str> for Format {
    fn eq(&self, other: &str) -> bool {
        matches!(
            (other, self),
            ("s8" | "S8", Format::S8)
                | ("u8" | "U8", Format::U8)
                | ("s16" | "S16", Format::S16)
                | ("u16" | "U16", Format::U16)
                | ("s32" | "S32", Format::S32)
                | ("u32" | "U32", Format::U32)
                | ("f32" | "F32", Format::F32)
                | ("f64" | "F64", Format::F64)
        )
    }
}

impl Format {
    /// Returns the supported `Format`s for a device's [`HwParams`]
    pub fn supported<'a>(params: &'a HwParams) -> impl Iterator<Item = Format> + 'a {
        [
            Format::S8,
            Format::U8,
            Format::S16,
            Format::U16,
            Format::S32,
            Format::U32,
            Format::F32,
            Format::F64,
        ]
        .into_iter()
        .filter(|&format| params.test_format(format.into()).is_ok())
    }
}

pub struct AudioRecorder<T> {
    pcm: PCM,
    channels: usize,
    rate: u32,
    buffer: Vec<T>,
    // audio: Audio,
}

impl<T: Copy + IoFormat + Num + ToPrimitive> AudioRecorder<T> {
    #[allow(clippy::missing_errors_doc)]
    pub fn new(
        name: impl AsRef<str>,
        channels: usize,
        rate: u32,
        format: Format,
        duration: f64,
    ) -> Result<Self, alsa::Error> {
        let pcm = PCM::new(name.as_ref(), Direction::Capture, false)?;
        {
            let hwp = HwParams::any(&pcm)?;
            #[allow(clippy::cast_possible_truncation)]
            hwp.set_channels(channels as u32)?;
            hwp.set_rate(rate, ValueOr::Nearest)?;
            hwp.set_format(format.into())?;
            hwp.set_access(Access::RWInterleaved)?;
            pcm.hw_params(&hwp)?;
        }
        pcm.start()?;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        Ok(AudioRecorder {
            pcm,
            channels,
            rate,
            // must be a full channels wide so only rate as float
            buffer: vec![T::zero(); (duration * f64::from(rate)) as usize * channels],
            // audio: Audio::empty(rate as F),
        })
    }

    // TODO borrow and reuse audio
    #[allow(clippy::missing_errors_doc)]
    pub fn record(&mut self) -> Result<Audio, alsa::Error> {
        _ = self.pcm.prepare();
        let io = self.pcm.io_checked()?;
        assert_eq!(
            io.readi(&mut self.buffer)?,
            self.buffer.len() / self.channels
        );
        Ok(Audio::from_interleaved(
            &self.buffer,
            self.channels,
            self.rate.into(),
        ))
    }
}

#[macro_export]
macro_rules! for_format {
    ($format:expr, $expr:expr) => {
        match $format {
            Format::S8 => {
                type FORMAT = i8;
                $expr
            }
            Format::U8 => {
                type FORMAT = u8;
                $expr
            }
            Format::S16 => {
                type FORMAT = i16;
                $expr
            }
            Format::U16 => {
                type FORMAT = u16;
                $expr
            }
            Format::S32 => {
                type FORMAT = i32;
                $expr
            }
            Format::U32 => {
                type FORMAT = u32;
                $expr
            }
            Format::F32 => {
                type FORMAT = f32;
                $expr
            }
            Format::F32 => {
                type FORMAT = f32;
                $expr
            }
            Format::F64 => {
                type FORMAT = f64;
                $expr
            }
            Format::F64 => {
                type FORMAT = f64;
                $expr
            }
        }
    };
}
