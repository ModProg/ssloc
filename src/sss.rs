use std::f64::consts::PI;
use std::ops::Add;

use itertools::Itertools;
use nalgebra::ComplexField;
use smart_default::SmartDefault;

use crate::{angles_to_unit_vec, Audio, Position, F};

/// Uses a Delay-and-Sum Beamformer to extract a single channel.
#[derive(SmartDefault, Clone, Debug, PartialEq)]
pub struct DelayAndSum {
    #[default = 343.0]
    pub speed_of_sound: F,
    #[default(Some(11))]
    pub filter: Option<usize>,
    pub mics: Vec<Position>,
}

impl DelayAndSum {
    /// Uses a Delay-and-Sum Beamformer to extract a single channel in the direction specified via `az` and `el`.
    pub fn delay_and_sum<'a>(
        &'a self,
        az: F,
        el: F,
        audio: &'a Audio,
    ) -> impl Iterator<Item = F> + 'a {
        let target = angles_to_unit_vec(az, el);

        let mics = self
            .mics
            .iter()
            .map(|m| (m - target).magnitude() / self.speed_of_sound)
            .collect_vec();

        let min = mics
            .iter()
            .copied()
            .min_by(F::total_cmp)
            .expect("at least one mic");

        let mics = mics.into_iter().map(|m| m - min).collect_vec();

        assert_eq!(audio.channels(), mics.len());

        // https://web.archive.org/web/20230605140150/http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
        let sub_sample: Box<dyn Fn(usize, F) -> Option<F>> = match self.filter {
            Some(filter_length) => Box::new(move |mic: usize, delay: F| {
                let filter_length = filter_length + filter_length % 2;
                let centre_tap = filter_length / 2;
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
                    .fold_options(0., F::add)
            }),
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            None => Box::new(|mic: usize, delay: F| audio.get(mic, delay.round() as usize)),
        };

        // we skip the first centre_tap samples, hopefully fine
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        (self.filter.unwrap_or_default() / 2 + 1..).map_while(move |i| {
            mics.iter()
                .enumerate()
                .map(|(mic, tau)| {
                    sub_sample(mic, i as F + *tau * audio.sample_rate)
                        .map(|s| s / audio.channels() as F)
                })
                .fold_options(0., F::add)
        })
    }
}
