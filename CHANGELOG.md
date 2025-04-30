# Changelog

All notable changes to **AnaFlow** will be documented in this file.

## [1.2.0] - 2025-05

See [#12](https://github.com/GeoStat-Framework/AnaFlow/pull/12)

### Enhancements
- added solutions based on the effective transmissivity for the ["Integral" variogram model](https://geostat-framework.readthedocs.io/projects/gstools/en/v1.7.0/api/gstools.covmodel.Integral.html):
  - `ext_thiem_int`: steady state solution
  - `ext_thiem_int_3d`: steady state solution incorporating vertical anisotropy
  - `ext_theis_int`: transient solution
  - `ext_theis_int_3d`: transient solution incorporating vertical anisotropy
- added `fix_T_well` and `fix_K_well` bool flag to transient heterogeneous solutions to be able to set if the well value for the effective transmissivity/conductivity should be determined from the limit (`True`) or from the upscaled value in the first ring segment (`False`, default)
  - **breaking**: the previous behavior was effectively this set to `True`, which for steep effective curves resulted in an increasing error in the effective head near the well

### Changes
- updated docs (use myst parser for markdown files, only generate html and pdf)
- updated CI (fixed artifacts up-/download action, see #14)
- use hatchling as build backend


## [1.1.0] - 2023-04

See [#11](https://github.com/GeoStat-Framework/AnaFlow/pull/11)

### Enhancements
- move to `src/` base package structure
- drop py36 support
- added archive support
- simplify documentation


## [1.0.1] - 2020-04-02

### Bugfixes
- `ModuleNotFoundError` not present in py35
- `np.asscalar` deprecated, use `array.item()`
- `CHANGELOG.md` links updated


## [1.0.0] - 2020-03-22

### Enhancements
- new TPL Solution
- new tools sub-module
- using pentapy to solve LES in laplace space
- solution for aparent transmissivity from neuman 2004
- added extended GRF model
- convenient functions for (inverse-)laplace transformation

### Bugfixes
- `lat_ext` was ignored by ext_theis_3d

### Changes
- py2.7 support dropped


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


[1.2.0]: https://github.com/GeoStat-Framework/AnaFlow/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/GeoStat-Framework/AnaFlow/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/GeoStat-Framework/AnaFlow/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GeoStat-Framework/AnaFlow/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/GeoStat-Framework/AnaFlow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GeoStat-Framework/AnaFlow/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/GeoStat-Framework/AnaFlow/compare/v0.1...v0.2.4
[0.1.0]: https://github.com/GeoStat-Framework/AnaFlow/releases/tag/v0.1
