#[cfg(feature = "wav")]
#[test]
fn main() {
    use float_cmp::assert_approx_eq;
    use itertools::Itertools;
    use nalgebra::vector;
    use ssloc::{Audio, MbssConfig, F};

    debug_assert!(false, "run in release mode");
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

    let test = MbssConfig::default()
        .create(mics)
        .locate_spec(&array, 2)
        .into_iter()
        .map(|(az, el)| (az.to_degrees(), el.to_degrees()))
        .collect_vec();
    assert_eq!(test.len(), 2);
    assert_approx_eq!(F, test[0].0, 45., epsilon = 0.0001);
    assert_approx_eq!(F, test[0].1, 45., epsilon = 0.0001);
    assert_approx_eq!(F, test[1].0, 135., epsilon = 0.0001);
    assert_approx_eq!(F, test[1].1, 0., epsilon = 0.0001);
}
