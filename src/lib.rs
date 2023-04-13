use core::fmt::Debug;
use itertools::Itertools;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::iter;
use std::ops::{Add, Sub};

use hound::{SampleFormat, WavReader};
use nalgebra::{matrix, vector, Complex, Const, DMatrix, Dyn, Matrix3xX, Vector3};
use ndarray::Array2;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_with::{serde_as, FromInto};
use time::{Date, PrimitiveDateTime, Time};

pub type F = f64;
pub type I = i64;
pub type C = Complex<F>;
pub type Position = Vector3<F>;

pub use matlab::MBSS_locate_spec;
mod matlab;

#[derive(Deserialize, Debug)]
struct PrimitiveDateTimeD {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f32,
}

impl From<PrimitiveDateTimeD> for PrimitiveDateTime {
    fn from(
        PrimitiveDateTimeD {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }: PrimitiveDateTimeD,
    ) -> Self {
        Self::new(
            Date::from_calendar_date(year, month.try_into().unwrap(), day).unwrap(),
            Time::from_hms_milli(
                hour,
                minute,
                second.round() as u8,
                ((second - second.round()) * 1000.) as u16,
            )
            .unwrap(),
        )
    }
}

fn deserialize_microphone_pos<'de, D>(d: D) -> Result<Vec<Position>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut vec = Vec::new();
    // TODO with prefix list
    // TODO remove serde_json
    for (idx, key, value) in HashMap::<String, serde_json::Value>::deserialize(d)?
        .into_iter()
        .filter_map(|(key, value)| -> Option<(usize, char, f64)> {
            let key = key.strip_prefix("mic")?;
            let (idx, field) = key.split_once('_')?;
            Some((
                idx.parse().ok()?,
                field.chars().next()?,
                value.as_f64()? as f64,
            ))
        })
    {
        let idx = idx - 1;
        if vec.len() <= idx {
            vec.extend(iter::repeat_with(Position::default).take(idx - vec.len() + 1))
        }
        match key {
            'x' => vec[idx].x = value,
            'y' => vec[idx].y = value,
            'z' => vec[idx].z = value,
            _ => todo!(),
        }
    }
    Ok(vec)
}

#[serde_as]
#[derive(Deserialize, Debug)]
struct ArrayPosition {
    #[serde_as(as = "FromInto<PrimitiveDateTimeD>")]
    #[serde(flatten)]
    time: PrimitiveDateTime,
    // #[serde(flatten)]
    // position: Position,
    #[serde(flatten, deserialize_with = "deserialize_microphone_pos")]
    microphones: Vec<Position>,
}
#[serde_as]
#[derive(Deserialize, Debug)]
struct SoundSourcePosition {
    #[serde_as(as = "FromInto<PrimitiveDateTimeD>")]
    #[serde(flatten)]
    time: PrimitiveDateTime,
    // #[serde(flatten)]
    // position: Position,
}

fn read_tsv<T: DeserializeOwned>(file: &[u8]) -> Vec<T> {
    let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(file);
    rdr.deserialize().collect::<Result<Vec<_>, _>>().unwrap()
}

pub struct Audio {
    data: Array2<F>,
    sample_rate: F,
}

impl Audio {
    pub fn channels(&self) -> usize {
        self.data.dim().0
    }
    pub fn samples(&self) -> usize {
        self.data.dim().1
    }

    pub fn from_file(arg: &str) -> Self {
        let mut array = File::open(arg).unwrap();
        let (header, data) = wav::read(&mut array).unwrap();
        let data = data.as_sixteen().unwrap();
        Self {
            sample_rate: header.sampling_rate as F,
            data: Array2::from_shape_fn(
                (
                    header.channel_count as usize,
                    data.len() / header.channel_count as usize,
                ),
                |(c, s)| data[c + s * header.channel_count as usize] as F / 2f64.powi(15),
            ),
        }
    }
}

impl<R: Read> From<WavReader<R>> for Audio {
    fn from(value: WavReader<R>) -> Self {
        let spec = value.spec();
        let channels = spec.channels.into();
        let duration = value.duration();
        let samples = value
            .into_samples()
            .map_ok(|v: i16| v as F / 2f64.powi(15))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        Self {
            sample_rate: spec.sample_rate as F,
            data: Array2::from_shape_fn((channels, duration as usize), |(c, s)| {
                samples[s * channels + c]
            }),
        }
    }
}
