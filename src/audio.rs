use std::error::Error;
use std::iter;
use std::path::Path;
use std::str::FromStr;

#[cfg(feature = "wav")]
pub use hound::SampleFormat as WavFormat;
use itertools::Itertools;
use ndarray::{Array2, Axis};
use num::{FromPrimitive, ToPrimitive};

use crate::F;

#[must_use]
pub struct Audio {
    pub(crate) sample_rate: F,
    pub(crate) data: Array2<F>,
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
        let reader = hound::WavReader::new(data).unwrap();
        let spec = reader.spec();
        match spec.sample_format {
            hound::SampleFormat::Float => {
                let data = reader.into_samples();
                Self::from_interleaved(
                    spec.sample_rate as F,
                    spec.channels as usize,
                    data.collect::<Result<Vec<f32>, _>>().unwrap(),
                )
            }
            hound::SampleFormat::Int => {
                // https://web.archive.org/web/20230605122301/https://gist.github.com/endolith/e8597a58bcd11a6462f33fa8eb75c43d
                let data = reader.into_samples();
                Self::from_interleaved(
                    spec.sample_rate as F,
                    spec.channels as usize,
                    data.map_ok(normalize_pcm_wav(spec.bits_per_sample))
                        .collect::<Result<Vec<F>, _>>()
                        .unwrap(),
                )
            }
        }
    }

    pub fn from_interleaved(
        sample_rate: F,
        channels: usize,
        data: impl IntoIterator<Item = impl Into<F>>,
    ) -> Self {
        let data = data.into_iter().map_into().collect_vec();
        Self {
            sample_rate,
            data: Array2::from_shape_fn((channels, data.len() / channels), |(c, s)| {
                data[c + s * channels].to_f64().unwrap()
            }),
        }
    }

    pub fn from_channels(
        sample_rate: F,
        channels: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<F>>>,
    ) -> Self {
        let mut channels = channels
            .into_iter()
            .map(IntoIterator::into_iter)
            .collect_vec();
        let mut channel = channels.len() - 1;
        Self::from_interleaved(
            sample_rate,
            channels.len(),
            iter::from_fn(|| {
                channel = (channel + 1) % channels.len();
                channels[channel].next()
            }),
        )
    }

    pub fn to_interleaved<T: FromPrimitive>(&self) -> impl Iterator<Item = T> + '_ {
        self.data
            .iter()
            .map(|&d| T::from_f64(d).expect("audio format can be converted"))
    }

    /// Produces wave data without header
    #[cfg(feature = "wav")]
    #[must_use]
    pub fn wav_data(&self, sample_format: WavFormat, bits_per_sample: u16) -> Vec<u8> {
        use std::io::Cursor;

        use hound::Sample;

        let mut out = Vec::new();
        let writer = &mut Cursor::new(&mut out);

        match sample_format {
            hound::SampleFormat::Float => {
                for sample in self.to_interleaved::<f32>() {
                    sample
                        .write_padded(writer, bits_per_sample, (bits_per_sample + 7) / 8)
                        .unwrap();
                }
            }
            hound::SampleFormat::Int => todo!(),
        }
        out
    }

    /// Produces wave data without header
    pub fn from_pcm_bytes(format: PcmFormat, sample_rate: F, channels: usize, data: &[u8]) -> Self {
        Self::from_interleaved(sample_rate, channels, format.from_data(data))
    }

    #[cfg(feature = "wav")]
    #[must_use]
    pub fn wav(&self, sample_format: WavFormat, bits_per_sample: u16) -> Vec<u8> {
        use std::io::Cursor;

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

    pub(crate) fn get(&self, mic: usize, sample: usize) -> Option<F> {
        self.data.get((mic, sample)).copied()
    }
}
/// Utility to normalize wav data
pub fn normalize_pcm_wav(bits_per_sample: u16) -> impl Fn(i32) -> F {
    match bits_per_sample {
        u @ 0..=8 => Box::new(move |s: i32| {
            (s - 2i32.pow(u as u32 - 1)) as F / (2f64.powi(u as i32 - 1) - 1.)
        }) as Box<dyn Fn(i32) -> F>,
        i => Box::new(move |s: i32| s as F / (2f64.powi(i as i32) - 1.)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    Float {
        bytes: u8,
        lower_endian: bool,
    },
    Int {
        signed: bool,
        bytes: u8,
        lower_endian: bool,
    },
}

impl FromStr for PcmFormat {
    type Err = Box<dyn Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let format = s.get(0..1).ok_or("format name empty")?.to_ascii_uppercase();
        let bytes = s[1..(s[1..]
            .chars()
            .position(|c| !c.is_ascii_digit())
            .unwrap_or(s.len() - 1))]
            .parse::<u8>()?
            / 8;
        let lower_endian = s.ends_with("LE");
        Ok(match format.as_str() {
            "F" => Self::Float {
                bytes,
                lower_endian,
            },
            i => Self::Int {
                signed: i == "S",
                bytes,
                lower_endian,
            },
        })
    }
}

impl PcmFormat {
    /// Amount of bytes `PcmFormat` has.
    #[must_use]
    pub fn bytes(self) -> u8 {
        match self {
            PcmFormat::Float { bytes, .. } | PcmFormat::Int { bytes, .. } => bytes,
        }
    }

    /// Returns whether `PcmFormat` uses lower endian.
    #[must_use]
    pub fn lower_endian(self) -> bool {
        match self {
            PcmFormat::Int { lower_endian, .. } | PcmFormat::Float { lower_endian, .. } => {
                lower_endian
            }
        }
    }

    /// Returns whether the `PcmFormat` is signed.
    #[must_use]
    pub fn signed(self) -> bool {
        !matches!(self, Self::Int { signed: false, .. })
    }

    pub fn from_data(self, data: &[u8]) -> impl Iterator<Item = F> + '_ {
        data.chunks_exact(self.bytes().into())
            .map(move |data| match self {
                PcmFormat::Float {
                    bytes,
                    lower_endian,
                } => match (bytes, lower_endian) {
                    (4, true) => f32::from_le_bytes(data.try_into().unwrap()).into(),
                    (4, false) => f32::from_be_bytes(data.try_into().unwrap()).into(),
                    (8, true) => f64::from_le_bytes(data.try_into().unwrap()),
                    (8, false) => f64::from_be_bytes(data.try_into().unwrap()),
                    (bytes, _) => {
                        panic!("invalid byte count {bytes} only 4 or 8 byte floats are supported")
                    }
                },
                PcmFormat::Int {
                    signed,
                    bytes,
                    lower_endian,
                } => {
                    let value = match (signed, lower_endian) {
                        (true, true) => i32::from_le_bytes(
                            [0, 1, 2, 3].map(|i| data.get(i).copied().unwrap_or_default()),
                        ) as F,
                        (true, false) => u32::from_le_bytes(
                            [3, 2, 1, 0].map(|i| if 4 - i > bytes { 0 } else { data[i as usize] }),
                        ) as F,
                        (false, true) => u32::from_le_bytes(
                            [0, 1, 2, 3].map(|i| data.get(i).copied().unwrap_or_default()),
                        ) as F,
                        (false, false) => u32::from_le_bytes(
                            [3, 2, 1, 0].map(|i| if 4 - i > bytes { 0 } else { data[i as usize] }),
                        ) as F,
                    };
                    let bytes = bytes as i32;
                    if signed {
                        value / (2f64.powi(bytes) - 1.)
                    } else {
                        (value - 2f64.powi(bytes - 1)) / (2f64.powi(bytes - 1) - 1.)
                    }
                }
            })
    }
}
