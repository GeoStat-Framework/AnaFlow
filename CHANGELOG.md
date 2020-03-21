# Changelog

All notable changes to **AnaFlow** will be documented in this file.


## [Unreleased]

### Enhancements
- new TPL Solution
- new tools module
- using pentapy to solve LES in laplace space
- solution for aparent transmissivity from neuman 2004
- added extended GRF model
- convenient functions for (inverse-)laplace transformation


## [0.4.0] - 2019-03-07

### Enhancements
- the output for transient tests now preserves the shapes of time and rad (better for plotting in 3D)
- the grf model is now the default way of calculating pumping tests in laplace space
- the grf_laplace routine was optimized to estimate the radius of the cone of depression
- the grf_laplace uses now the pentapy solver, so we get rid of the umf_pack dependency
- grf_model and grf_disk are now part of the standard routines

### Changes
- the input for transient tests changed from "rad, time" to "time, rad" as order of input (in line with output format)

### Bugfixes
- multiple minor bugfixes


## [0.3.0] - 2019-02-28

### Enhancements
- GRF model added
- new documetation
- added examples
- code restructured

### Changes

### Bugfixes


## [0.2.4] - 2018-04-26

### Enhancements
- Released on PyPI


## [0.1.0] - 2018-01-05

First release of AnaFlow.
Containing:
- thiem - Thiem solution for steady state pumping
- theis - Theis solution for transient pumping
- ext_thiem2D - extended Thiem solution in 2D
- ext_theis2D - extended Theis solution in 2D
- ext_thiem3D - extended Thiem solution in 3D
- ext_theis3D - extended Theis solution in 3D
- diskmodel - Solution for a diskmodel
- lap_transgwflow_cyl - Solution for a diskmodel in laplace inversion


[Unreleased]: https://github.com/GeoStat-Framework/gstools/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/GeoStat-Framework/gstools/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GeoStat-Framework/gstools/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/GeoStat-Framework/gstools/compare/v0.1...v0.2.4
[0.1.0]: https://github.com/GeoStat-Framework/gstools/releases/tag/v0.1
