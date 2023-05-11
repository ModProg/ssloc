use std::fmt::Debug;
use std::path::PathBuf;
use std::{fs, iter};

use alsa::device_name::{Hint, HintIter};
use alsa::pcm::{Access, HwParams, IoFormat, PCM};
use alsa::{Direction, Error, ValueOr};
use anyhow::Context;
use clap::{Parser, Subcommand};
use confique::toml::{template, FormatOptions};
use confique::Config as _;
use derive_more::{
    Add, AddAssign, AsMut, AsRef, Deref, DerefMut, From, Into, Mul, MulAssign, Sub, Sum,
};
use itertools::Itertools;
use num::{Num, NumCast, ToPrimitive};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, FromInto};
use ssloc::{for_format, Audio, AudioRecorder, Format, MbssConfig, F};
use unidirs::{Directories, UnifiedDirs, Utf8PathBuf};

type Result<T = (), E = anyhow::Error> = std::result::Result<T, E>;

const NAME: &str = "ros-ssl";
const CONFIG_NAME: &str = "config.toml";
const GLOBAL: &str = "/etc/ros-ssl/config.toml";

#[serde_as]
#[derive(
    Sub,
    AsRef,
    AsMut,
    From,
    Into,
    Deref,
    DerefMut,
    Add,
    Mul,
    Sum,
    AddAssign,
    MulAssign,
    Serialize,
    Deserialize,
    Clone,
)]
#[serde(transparent)]
pub struct Position(#[serde_as(as = "FromInto<[F;3]>")] ssloc::Position);
impl Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?}, {:?})", &self.0.x, &self.0.y, &self.0.z)
    }
}

fn start_capture(device: &str, channels: usize, format: Format, rate: u32) -> Result<PCM, Error> {
    let pcm = PCM::new(device, Direction::Capture, false)?;
    {
        let hwp = HwParams::any(&pcm)?;
        hwp.set_channels(channels as u32)?;
        hwp.set_rate(rate, ValueOr::Nearest)?;
        hwp.set_format(format.into())?;
        hwp.set_access(Access::RWInterleaved)?;
        pcm.hw_params(&hwp)?;
    }
    pcm.start()?;
    Ok(pcm)
}

// Calculates RMS (root mean square) as a way to determine volume
#[must_use]
fn rms<T: Num + ToPrimitive + Copy>(buf: &[T], channel: usize, channels: usize) -> f64 {
    if buf.is_empty() {
        return 0f64;
    }
    let mut sum = 0f64;
    for &x in buf.iter().skip(channel).step_by(channels) {
        sum += <f64 as NumCast>::from(x).unwrap().powi(2);
    }
    let r = (sum / (buf.len() as f64)).sqrt();
    // Convert value to decibels
    20.0 * (r / (i16::MAX as f64)).log10()
}

fn read_loop<S: IoFormat + Num + ToPrimitive>(pcm: &PCM, channels: usize) -> Result<(), Error> {
    const SAMPLES: usize = 2usize.pow(13);
    dbg!(SAMPLES);
    let io = pcm.io_checked()?;
    // Buffer needs space for SAMPLES for every channel
    let mut buf: Vec<S> = vec![S::zero(); SAMPLES * channels];
    loop {
        // Block while waiting for 8192 samples to be read from the device.
        assert_eq!(io.readi(&mut buf)?, SAMPLES);
        print!("RMS:");
        for channel in 0..channels {
            // Buffer is interleaved
            print!(" {:.1}", rms(&buf, channel, channels));
        }
        println!(" dB")
    }
}

#[derive(Parser)]
struct Clap {
    #[clap(subcommand)]
    subcommand: Command,
    /// Use a custom config
    #[arg(long, short)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Prints available capture devices
    Devices,
    /// Logs the volume for each channel of a audio device
    Test {
        device: String,
        #[arg(long, short, default_value = "1")]
        channels: usize,
        #[arg(long, short, default_value = "s16")]
        format: Format,
        #[arg(long, short, default_value = "44100")]
        rate: u32,
    },
    #[command(subcommand)]
    Config(ConfigCommand),
    /// Prints the angles found through the sound source localization
    PrintSsl {
        /// Print the angles in degrees
        #[arg(long, short)]
        degrees: bool,
        #[arg(long, short)]
        file: Option<PathBuf>,
        #[arg(long, short)]
        #[cfg(feature = "image")]
        image: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Prints current configuration
    Show,
    /// Creates a config file
    Create {
        /// Path to create the config at
        #[arg(long, short, default_value_t = config_file())]
        path: Utf8PathBuf,
        /// System Config
        ///
        /// Creates the config on sytem level `/etc/ros-ssl/config.toml`
        #[arg(long, short)]
        system: bool,
    },
}

#[derive(confique::Config, Debug)]
pub struct Config {
    #[config(default = [[1, 1, 1]])]
    pub mics: Vec<Position>,
    // TODO create config with defaults
    pub mbss: Option<MbssConfig>,
    #[config(default = 1)]
    pub max_sources: usize,
    /// To get a list of available devices run `ros-ssl devices`
    #[config(default = "default")]
    pub alsa_name: String,
    #[config(default = 44100)]
    pub rate: u32,
    #[config(default = "s16")]
    pub format: Format,
    /// Duration of recording for each localisation
    #[config(default = 1.0)]
    pub localisation_frame: F,
}

fn config_file() -> Utf8PathBuf {
    UnifiedDirs::simple("de", "modprog", NAME)
        .with_env()
        .with_username()
        .build()
        .unwrap()
        .config_dir()
        .join(CONFIG_NAME)
}

fn main() -> Result {
    let Clap {
        subcommand,
        config: config_path,
    } = Clap::parse();
    let mut config = Config::builder().env();
    if let Some(config_path) = config_path {
        config = config.file(config_path);
    };
    let config = config.file(config_file()).file(GLOBAL).load()?;
    match subcommand {
        Command::Devices => {
            for ref hint @ Hint {
                ref name,
                ref desc,
                direction,
            } in HintIter::new_str(None, "pcm")?.chain(iter::once(Hint {
                name: Some("default".into()),
                desc: Some("System Default".into()),
                direction: Some(Direction::Capture),
            })) {
                if matches!(direction, Some(Direction::Capture)) {
                    let desc = desc.as_deref().unwrap_or_default().replace('\n', " ");
                    if let Some(name) = name {
                        println!("{:-<16}", "");
                        println!("{desc}:");
                        println!("    Name: {name:?}");
                        match PCM::new(name, Direction::Capture, false) {
                            Ok(pcm) => match HwParams::any(&pcm) {
                                Ok(params) => {
                                    if let Ok(channels) = params.get_channels() {
                                        println!("    Channels: {channels}");
                                    } else if let (Ok(from), Ok(to)) =
                                        (params.get_channels_min(), params.get_channels_max())
                                    {
                                        println!("    Channels: {from}..={to}");
                                    }
                                    if let Ok(rate) = params.get_rate() {
                                        println!("    Rate: {rate}");
                                    } else if let (Ok(from), Ok(to)) =
                                        (params.get_rate_min(), params.get_rate_max())
                                    {
                                        println!("    Rate: {from}..={to}");
                                    }
                                    println!(
                                        "    Format: {}",
                                        Format::supported(&params).join(", ")
                                    );
                                }
                                Err(err) => println!("    {err}"),
                            },
                            Err(err) => println!("    {err}"),
                        }
                    } else {
                        eprintln!("Unnamed audio device: {hint:?}");
                    }
                }
            }
        }
        Command::Test {
            device,
            channels,
            format,
            rate,
        } => {
            let capture = start_capture(&device, channels, format, rate)?;
            for_format!(format, read_loop::<FORMAT>(&capture, channels)?);
        }
        Command::Config(ConfigCommand::Show) => {
            println!("{config:#?}");
        }
        Command::Config(ConfigCommand::Create { mut path, system }) => {
            let config = template::<Config>(FormatOptions::default());
            if system {
                fs::write(GLOBAL, config).context(format!("writing the config to {GLOBAL}"))?;
                println!("Created config at {GLOBAL}");
            } else {
                if path.is_dir() {
                    path = path.join(CONFIG_NAME);
                } else if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).context(format!("creating folder at {parent}"))?;
                } else {
                    unreachable!("{path} is root without being a dir")
                }
                fs::write(&path, config).context(format!("writing the config to {path}"))?;
                println!("Created config at {path}");
            }
        }
        Command::PrintSsl {
            degrees,
            file,
            #[cfg(feature = "image")]
            image,
        } => {
            let mbss = config.mbss.unwrap_or_default().create(config.mics.clone());
            eprintln!(
                "Array zentroid: {:?}",
                Position::from(mbss.array_centroid())
            );
            eprintln!(
                "{}",
                (0..config.max_sources)
                    .map(|i| format!("{:>21}\t", format!("source {i:3}")))
                    .collect::<String>()
            );
            eprintln!(
                "{}",
                format!("{:>10} {:>10}\t", "azimuth", "elevation").repeat(config.max_sources)
            );
            if let Some(file) = file {
                let spectrum = mbss.analyze_spectrum(&Audio::from_file(file));
                #[cfg(feature = "image")]
                if let Some(mut image) = image {
                    if image.is_dir() {
                        image = image.join("spec.png");
                    }
                    ssloc::spec_to_image(spectrum.view())
                        .save_with_format(&image, image::ImageFormat::Png)
                        .with_context(|| format!("writing spectrum to {}", image.display()))?;
                }
                let sources = mbss.find_sources(spectrum.view(), config.max_sources);
                for (az, el, _) in sources {
                    if degrees {
                        print!("{:9.0}° {:9.0}°\t", az.to_degrees(), el.to_degrees())
                    } else {
                        print!("{az:10.7} {el:10.7}\t")
                    }
                }
                return Ok(());
            };
            for_format!(config.format, {
                let mut recorder = AudioRecorder::<FORMAT>::new(
                    config.alsa_name,
                    config.mics.len(),
                    config.rate,
                    config.format,
                    config.localisation_frame,
                )?;
                loop {
                    let sources = mbss.locate_spec(&recorder.record()?, config.max_sources);
                    for (az, el, _) in sources {
                        if degrees {
                            print!("{:9.0}° {:9.0}°\t", az.to_degrees(), el.to_degrees())
                        } else {
                            print!("{az:10.7} {el:10.7}\t")
                        }
                    }
                    println!()
                }
            });
        }
    }
    Ok(())
}
