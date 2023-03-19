use core::fmt::Debug;
use std::collections::HashMap;
use std::fs::File;
use std::iter;

use fundsp::hacker32::*;
use hound::WavReader;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_with::{serde_as, FromInto};
use time::{Date, PrimitiveDateTime, Time};

#[derive(Deserialize, Default)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
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
        .filter_map(|(key, value)| -> Option<(usize, char, f32)> {
            let key = key.strip_prefix("mic")?;
            let (idx, field) = key.split_once('_')?;
            Some((
                idx.parse().ok()?,
                field.chars().next()?,
                value.as_f64()? as f32,
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
    let sound_source_positions: Vec<SoundSourcePosition> = read_tsv(include_bytes!(
        "../references/LOCATA/dev/task1/recording1/benchmark2/position_source_loudspeaker1.txt"
    ));
    assert_eq!(array_positions.len(), sound_source_positions.len());
    assert_eq!(array_positions[0].time, sound_source_positions[0].time);

    let source = Wave32::load(
        "references/LOCATA/dev/task1/recording1/benchmark2/audio_source_loudspeaker1.wav",
    )
    .expect("Could not load source audio");
    assert_eq!(source.channels(), 1);

    let array = WavReader::open(
        "references/LOCATA/dev/task1/recording1/benchmark2/audio_array_benchmark2.alt.wav",
    )
    .unwrap();
    assert_eq!(
        array.spec().channels as usize,
        array_positions[0].microphones.len()
    );
}
