use core::fmt::Debug;
use std::collections::HashMap;
use std::iter;
use std::ops::Add;

use hound::{SampleFormat, WavReader};
use itertools::Itertools;
use nalgebra::{matrix, Const, DMatrix, Dyn, Matrix3xX};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_with::{serde_as, FromInto};
use time::{Date, PrimitiveDateTime, Time};

use crate::matlab::MBSS_locate_spec;

mod matlab;

#[derive(Deserialize, Default, Clone, Copy)]
struct Position {
    x: f64,
    y: f64,
    z: f64,
}

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}

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
    #[serde(flatten)]
    position: Position,
    #[serde(flatten, deserialize_with = "deserialize_microphone_pos")]
    microphones: Vec<Position>,
}
#[serde_as]
#[derive(Deserialize, Debug)]
struct SoundSourcePosition {
    #[serde_as(as = "FromInto<PrimitiveDateTimeD>")]
    #[serde(flatten)]
    time: PrimitiveDateTime,
    #[serde(flatten)]
    position: Position,
}

fn read_tsv<T: DeserializeOwned>(file: &[u8]) -> Vec<T> {
    let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(file);
    rdr.deserialize().collect::<Result<Vec<_>, _>>().unwrap()
}

fn main() {
    let array_positions: Vec<ArrayPosition> = read_tsv(include_bytes!(
        "../references/LOCATA/dev/task1/recording1/benchmark2/position_array_benchmark2.txt"
    ));
    // let sound_source_positions: Vec<SoundSourcePosition> = read_tsv(include_bytes!(
    //     "../references/LOCATA/dev/task1/recording1/benchmark2/position_source_loudspeaker1.txt"
    // ));
    // assert_eq!(array_positions.len(), sound_source_positions.len());
    // assert_eq!(array_positions[0].time, sound_source_positions[0].time);

    // let mics = Matrix3xX::from_row_iterator(
    //     array_positions[0].microphones.len(),
    //     array_positions[0]
    //         .microphones
    //         .iter()
    //         .map(|&p|p + array_positions[0].position)
    //         .flat_map(|Position { x, y, z }| [x, y, z]),
    // );
    #[rustfmt::skip]
    let mics = [
        Position{ x:  0.0370, y:  0.0560, z: -0.0380 },
        Position{ x: -0.0340, y:  0.0560, z:  0.0380 },
        Position{ x: -0.0560, y:  0.0370, z: -0.0380 },
        Position{ x: -0.0560, y: -0.0340, z:  0.0380 },
        Position{ x: -0.0370, y: -0.0560, z: -0.0380 },
        Position{ x:  0.0340, y: -0.0560, z:  0.0380 },
        Position{ x:  0.0560, y: -0.0370, z: -0.0380 },
        Position{ x:  0.0560, y:  0.0340, z:  0.0380 },
    ];
    //  .transpose();
    // let mics = Matrix3xX::from_iterator(8, mics.into_iter().copied());

    // let source = Wave32::load(
    //     "references/LOCATA/dev/task1/recording1/benchmark2/audio_source_loudspeaker1.wav",
    // )
    // .expect("Could not load source audio");
    // assert_eq!(source.channels(), 1);

    // let mut array = WavReader::open(
    //     "references/LOCATA/dev/task1/recording1/benchmark2/audio_array_benchmark2.alt.wav",
    // )
    let mut array = WavReader::open(
        "references/mbss_locate/v2.0/examples/example_1/wav files/male_female_mixture.wav",
    )
    .unwrap();
    assert_eq!(array.spec().channels as usize, mics.ncols());
    assert_eq!(array.spec().sample_format, SampleFormat::Int);

    let duration = dbg!(array.duration()) as usize;
    // let duration = 100_000;

    let x = DMatrix::from_row_iterator(
        mics.ncols(),
        duration,
        array
            .samples::<i32>()
            .take(duration * mics.ncols())
            .map(|v| (v.unwrap() as f64) / i32::MAX as f64),
    );

    let test = MBSS_locate_spec(x, 1. / (array.spec().sample_rate as f64), mics, 2);
    println!("{test:?}");
}
