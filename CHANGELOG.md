# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

[unreleased]: https://github.com/ModProg/ssloc/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ModProg/ssloc/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ModProg/ssloc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ModProg/ssloc/releases/tag/v0.1.0
