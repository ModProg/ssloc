#![cfg(feature = "wav")]
use float_cmp::assert_approx_eq;
use itertools::Itertools;
use ssloc::{Audio, F};
#[test]
fn round_trip_pcm() {
    let expected = Audio::from_file("tests/input/male_female_mixture.wav");
    let actual = expected
        .to_interleaved()
        .flat_map(f32::to_le_bytes)
        .collect_vec();
    let actual = Audio::from_pcm_bytes(
        "F32LE".parse().unwrap(),
        expected.sample_rate(),
        expected.channels(),
        &actual,
    );
    assert_approx_eq!(&Audio, &actual, &expected, epsilon = 0.0001);
}

#[test]
fn round_trip_interleaved() {
    let expected = Audio::from_file("tests/input/male_female_mixture.wav");
    let actual = expected.to_interleaved::<F>();
    let actual = Audio::from_interleaved(expected.sample_rate(), expected.channels(), actual);
    assert_approx_eq!(&Audio, &actual, &expected, epsilon = 0.0001);
}
