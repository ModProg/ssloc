use std::f64::consts::PI;
use std::ops::Add;

use itertools::Itertools;
use nalgebra::ComplexField;

use crate::{angles_to_unit_vec, Audio, Position, F};

/// Uses a Delay-and-Sum Beamformer to extract a single channel in the direction specified via `az`
/// and `el`.
pub fn delay_and_sum(
    az: F,
    el: F,
    audio: &Audio,
    mics: impl IntoIterator<Item = impl Into<Position>>,
    speed_of_sound: F,
) -> impl Iterator<Item = F> + '_ {
    let target = angles_to_unit_vec(az, el);
    println!("{:?}", (target));

    let mics = mics
        .into_iter()
        .map(Into::into)
        .map(|m| (m - target).magnitude() / speed_of_sound)
        .collect_vec();

    println!("{:?}", (mics));
    let min = mics
        .iter()
        .copied()
        .min_by(F::total_cmp)
        .expect("at least one mic");

    println!("{:?}", (min));
    let mics = mics.into_iter().map(|m| m - min).collect_vec();
    println!("{:?}", (mics));

    assert_eq!(audio.channels(), mics.len());

    // https://web.archive.org/web/20230605140150/http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
    let filter_length = 11;
    let centre_tap = filter_length / 2;
    let sub_sample = move |mic: usize, delay: F| {
        Some(
            (0..filter_length)
                .map(move |t| {
                    let x = t as F - (delay - delay.floor());
                    let sinc = (PI * (x - centre_tap as F)).sinc();
                    let window = 0.54 - 0.46 * (2. * PI * (x + 0.5) / filter_length as F).cos();
                    let tap_weight = window * sinc;

                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    {
                        Some(audio.get(mic, t + delay as usize - centre_tap)? * tap_weight)
                    }
                })
                .fold_options(0., F::add)?
                / audio.channels() as F,
        )
    };

    // we skip the first filter_length samples, hopefully fine
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    (filter_length..).map_while(move |i| {
        mics.iter()
            .enumerate()
            .map(|(mic, tau)| sub_sample(mic, i as F + *tau * audio.sample_rate))
            // .map(|(mic, tau)| {
            //     audio
            //         .get(mic, i + dbg!((tau * audio.sample_rate).round()) as usize)
            //         .map(|s| s / mics.len() as F)
            // })
            .fold_options(0., F::add)
    })
}
