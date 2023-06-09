# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Added documentation for many items.

### Fixed
- incorrect calculation of angular distance in `angular_distance`
- `for_format!` didn't work without importing `ssloc::Format`

### Changed
- **BREAKING CHANGE** renamed `spectrum` to `intensities` in many function names.

## [0.4.1] - 2023-07-04
### Fixed
- removed panic due to division through zero

## [0.4.0] - 2023-07-04
### Added
- `Direction` a struct holding a direction in azimuth and elevation
- `DelayAndSum::expected_len()`
- `Audio::from_pcm_data()`
- `ApproxEq` implementation for Audio

### Changed
- **BREAKING CHANGE** replaced many uses of `(azimuth, elevation)` with new `Direction` struct
- **BREAKING CHANGE** renamed `DelayAndSum::delay_and_sum()` to `beam_form()`
- Audio recorder no longer panics if unable to record full length, returns shorter `Audio` instead.

## [0.3.0] - 2023-06-15
### Added
- `Audio::sample_rate()`
- `DelayAndSum` sound source separation.
- `Audio::wav_data()`

### Changed
- **BREAKING CHANGE** `Audio::wav()` now takes `WavFormat` and `bits_per_sample: u16`.
- **BREAKING CHANGE** `Audio::from_interleaved()` now takes `impl IntoIterator<Item = impl Into<f64>>` and changed order of inputs.
- replaced `wav` with `hound`.

### CLI
- added `sss` subcommand supporting `delay-and-sum` algorithm for sound source separation.

## [0.2.0] - 2023-06-01
### Added
- `mbss::angular_distance`

## [0.1.1] - 2023-05-31
### Fixed
- `Pooling::Sum`

## [0.1.0] - 2023-05-14
INITIAL RELEASE

[unreleased]: https://github.com/ModProg/ssloc/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/ModProg/ssloc/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/ModProg/ssloc/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ModProg/ssloc/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ModProg/ssloc/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ModProg/ssloc/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ModProg/ssloc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ModProg/ssloc/releases/tag/v0.1.0
