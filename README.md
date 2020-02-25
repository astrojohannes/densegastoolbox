# densegastoolbox

Aim: Calculate density and temperature from observed molecular lines.

Method: Minimize observed line ratios against radiative transfer models.
Our models assume that the molecular emission lines emerge from a
multi-density medium rather than from a single density alone.

Results: Using an ascii table of observed molecular intensities [K km/s],
the results (mass-weighted mean density, temperature and width of the density
distribution) are saved in an output ascii file. Furthermore, diagnostic plots
are created to assess the quality of the fit/derived parameters.

Current Version 1.0 notes:
* The initial release contains models for the following lines:
  12CO (1-0), 12CO (2-1), 12CO (3-2), 13CO (1-0), HCN (1-0), HNC (1-0) and
  HCO+ (1-0)

* See example.py for how to use "Dense Gas Toolbox". It's easy!

* Needs Python 2.7 (will be upgraded to 3.X).
