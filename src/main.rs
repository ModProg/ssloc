use alsa::pcm::*;
use alsa::{Direction, Error, ValueOr};
use itertools::Itertools;
use nalgebra::vector;
use ssl::{Audio, MbssConfig};

fn start_capture(device: &str) -> Result<PCM, Error> {
    let pcm = PCM::new(device, Direction::Capture, false)?;
    {
        let hwp = HwParams::any(&pcm)?;
        hwp.set_channels(1)?;
        hwp.set_rate(44100, ValueOr::Nearest)?;
        hwp.set_format(Format::s16())?;
        hwp.set_access(Access::RWInterleaved)?;
        pcm.hw_params(&hwp)?;
    }
    pcm.start()?;
    Ok(pcm)
}

// Calculates RMS (root mean square) as a way to determine volume
fn rms(buf: &[i16]) -> f64 {
    if buf.len() == 0 {
        return 0f64;
    }
    let mut sum = 0f64;
    for &x in buf {
        sum += (x as f64) * (x as f64);
    }
    let r = (sum / (buf.len() as f64)).sqrt();
    // Convert value to decibels
    20.0 * (r / (i16::MAX as f64)).log10()
}

fn read_loop(pcm: &PCM) -> Result<(), Error> {
    let io = pcm.io_i16()?;
    let mut buf = [0i16; 8192];
    loop {
        // Block while waiting for 8192 samples to be read from the device.
        assert_eq!(io.readi(&mut buf)?, buf.len());
        let r = rms(&buf);
        println!("RMS: {:.1} dB", r);
    }
}

fn main() {
    // The "default" device is usually directed to the sound server process,
    // e g PulseAudio or PipeWire.
    let capture = start_capture("default").unwrap();
    read_loop(&capture).unwrap();
}

// fn main() {
//     // #[rustfmt::skip]
//     // let mics = [
//     //     vector![  0.0370,  0.0560, -0.0380 ],
//     //     vector![ -0.0340,  0.0560,  0.0380 ],
//     //     vector![ -0.0560,  0.0370, -0.0380 ],
//     //     vector![ -0.0560, -0.0340,  0.0380 ],
//     //     vector![ -0.0370, -0.0560, -0.0380 ],
//     //     vector![  0.0340, -0.0560,  0.0380 ],
//     //     vector![  0.0560, -0.0370, -0.0380 ],
//     //     vector![  0.0560,  0.0340,  0.0380 ],
//     // ];
//     //
//     // let array = Audio::from_file(
//     //     "references/mbss_locate/v2.0/examples/example_1/wav files/male_female_mixture.wav",
//     // );
//     // assert_eq!(array.channels(), mics.len());
//     //
//     // let test = MbssConfig::default()
//     //     .create(mics)
//     //     .locate_spec(&array, 2)
//     //     .into_iter()
//     //     .map(|(az, el)| (az.to_degrees(), el.to_degrees()))
//     //     .collect_vec();
//     // println!("{test:?}");
// }
