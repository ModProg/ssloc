use itertools::Itertools;
use ssl::{Audio, MBSS_locate_spec};

use nalgebra::vector;

fn main() {
    #[rustfmt::skip]
    let mics = [
        vector![  0.0370,  0.0560, -0.0380 ],
        vector![ -0.0340,  0.0560,  0.0380 ],
        vector![ -0.0560,  0.0370, -0.0380 ],
        vector![ -0.0560, -0.0340,  0.0380 ],
        vector![ -0.0370, -0.0560, -0.0380 ],
        vector![  0.0340, -0.0560,  0.0380 ],
        vector![  0.0560, -0.0370, -0.0380 ],
        vector![  0.0560,  0.0340,  0.0380 ],
    ];

    let array = Audio::from_file(
        "references/mbss_locate/v2.0/examples/example_1/wav files/male_female_mixture.wav",
    );
    assert_eq!(array.channels(), mics.len());

    let test = MBSS_locate_spec(array, &mics, 2)
        .into_iter()
        .map(|(az, el)| (az.to_degrees(), el.to_degrees()))
        .collect_vec();
    println!("{test:?}");
}
