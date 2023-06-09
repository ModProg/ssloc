#[cfg(feature = "wav")]
#[test]
fn mbss_locate_example1() {
    use float_cmp::approx_eq;
    use itertools::Itertools;
    use nalgebra::vector;
    use ssloc::mbss::Pooling;
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

    let array = Audio::from_file("tests/input/male_female_mixture.wav");
    assert_eq!(array.channels(), mics.len());

    for config in [MbssConfig::default(), MbssConfig {
        pooling: Pooling::Sum,
        ..Default::default()
    }] {
        eprintln!("Config: {config:?}");
        let test = config
            .create(mics)
            .locate_spec(&array, 2)
            .into_iter()
            .map(|v| dbg!(v))
            .map(|(dir, _)| (dir.azimuth.to_degrees(), dir.elevation.to_degrees()))
            .collect_vec();
        assert_eq!(test.len(), 2);
        assert!(
            approx_eq!(F, test[0].0, 45., epsilon = 2.)
                && approx_eq!(F, test[0].1, 45., epsilon = 2.)
                && approx_eq!(F, test[1].0, 135., epsilon = 2.)
                && approx_eq!(F, test[1].1, 0., epsilon = 2.)
                || approx_eq!(F, test[1].0, 45., epsilon = 2.)
                    && approx_eq!(F, test[1].1, 45., epsilon = 2.)
                    && approx_eq!(F, test[0].0, 135., epsilon = 2.)
                    && approx_eq!(F, test[0].1, 0., epsilon = 2.),
            "expected [(45., 45.), (135., 0.)] or [(135., 0.), (45., 45.)] got {test:?}"
        )
    }
}
