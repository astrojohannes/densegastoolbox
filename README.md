# Dense Gas Toolbox #
DOI: 10.5281/zenodo.3686329

# Aim
Calculate density and temperature from observed molecular emission lines,
using radiative transfer models.

# Method
Our models assume that the molecular emission lines emerge from a
multi-density medium rather than from a single density alone.
The density distribution is assumed to be log-normal or log-normal with
a power-law tail.
The parameters (density, temperature and the width of density distribution)
are inferred using Bayesian statistics, i.e. Markov chain Monte Carlo (MCMC).

# Results
Given an ascii table of observed molecular intensities [K km/s],
the results (mass-weighted mean density, temperature and width of the density
distribution) are saved in an output ascii file. Furthermore, diagnostic plots
are created to assess the quality of the fit/derived parameters.

---

# VERSION HISTORY

- May 22, 2024 | Version 1.7 
   * This is the latest release that is based on the fixed optical depth models (as used since version 1.3)

   * Updated code to account for deprecation, future and syntax warnings

   * The following molecular transitions are covered:
        CO (1-0)      CO (2-1)        CO (3-2)
        HCN (1-0)     HCN (2-1)       HCN (3-2)
        HCOP (1-0)    HCOP (2-1)      HCOP (3-2)
        HNC (1-0)     HNC (2-1)       HNC (3-2)
        13CO (1-0)    13CO (2-1)      13CO (3-2)
        C18O (1-0)    C18O (2-1)      C18O (3-2)
        C17O (1-0)    C17O (2-1)      C17O (3-2)
        CS (1-0)      CS (2-1)        CS (3-2)

   * Temperatures range from 10 to 50 K (in steps of 5K)

   * The widths of the density distributions range from 0.2 to 0.9dex (in steps of 0.1dex)

   * See "example.py" for how to use the Dense Gas Toolbox. It's easy!

---

- May 21, 2024 | Version 1.6

   * Bugfixed corner plot axes limits

   * Added debugging statements

   * Added INSTALL file

---

- May 23, 2022 | Version 1.5 (minor update):

   * Update: User may now also enter/change the number of MCMC simulations to run

   * Bugfix: In cases where sampler did not converge (Tau is infinite or log-probability
     is neg. infinity) previous versions crashed with "ValueError: cannot convert float
     NaN to integer" when trying to produce corner plots. Now, the program continues
     execution and shows one of the following Warnings: "Warning: Tau is NaN, corner plot
     cannot be created!" or "Warning: MCMC did not converge, you may try to increase the
     number of simulations (nsims)!"

---

- Feb 27, 2021 | Version 1.4 (minor update):

   * Bugfix: Fixing import of CS model grid

   * Update: Code updated to remove deprecation warnings

---

- Feb 26, 2021 | Version 1.3 (major update):
   
   * New: The user may optionally infer the parameters (density, temperature, width of
     density distribution) via application of the MCMC method.
   
   * New: Diagnosis plots (corner plots) are produced when MCMC method is used.
     
   * Update: Code updated to Python 3.X
      
   * Update: Re-calculation of models, now including the following transitions:
     12CO (up to J=3), 13CO (up to J=3), C18O (up to J=3), C17O (up to J=3),
     HCN (up to J=3), HCO+ (up to J=3), HNC (up to J=3) and CS (up to J=3)

---

- Mar 31, 2020 | Version 1.2 (major update):

   * New: An online version is now available at:

                http://www.densegastoolbox.com

   * New: The models can be explored using an interactive application at:

                http://www.densegastoolbox.com/explorer

   * Update: Model fit parameters are: density (mass-weighted), temperature and width
     of the distribution. The temperature and NOW ALSO WIDTH can be fixed.

   * Update: Re-calculation of models, now including the following transitions:
     12CO (up to J=3), 13CO (up to J=3), C18O (up to J=3), C17O (up to J=3),
     HCN (up to J=3), HCO+ (up to J=3) and HNC (up to J=3)

   * Update: The one-zone model grids are now more extended with H_2 densities
     between 10^-2 and 10^8 cm^-3.

   * Update: Diagnosis plots improved: (n,T) vs. chi2 and (n,width) vs. chi2

   * See "example.py" for how to use the Dense Gas Toolbox. It's easy!

   * Still needs Python 2.7 (will be upgraded to 3.X).

---

- Feb 28, 2020 | Version 1.1 (minor update):

   * This release now includes the data table ("ascii_galaxy.txt") used by "example.py".

---

- Feb 25, 2020 | Version 1.0:

   * The initial release contains models for the following lines:
     12CO (1-0), 12CO (2-1), 12CO (3-2), 13CO (1-0), HCN (1-0), HNC (1-0) and
     HCO+ (1-0)

   * abundances and optical depths are fixed based on observations of the
     EMPIRE survey

   * Two density distributions (PDFs) are available: lognorm and lognorm+power law

   * Model fit parameters are: density (mass-weighted), temperature and width
     of the distribution. The temperature can be fixed.

   * Emissivities per density bin are calculated using RADEX, i.e. LVG method
     for an expanding sphere. These one-zone model grids are limited to H_2
     densities between 10 and 10^8 cm^-3.

   * The total line intensity is found from summation of the emissivities per H_2
     molecule along the gas density distribution (and multiplication with
     column per linewidth and abundance). For some models, at the very low and
     very high density ends, the extent of the PDF exceeds the one-zone density grid
     limits (10-10^8 cm^-3). In these regimes (where emissivities are very low anyway
     for the molecules under consideration), the emissivities are set constant to
     the grid limit value.

   * See "example.py" for how to use the Dense Gas Toolbox. It's easy!

   * Needs Python 2.7 (will be upgraded to 3.X).

   * Depends on the following Python packages:
     numpy, matplotlib, pylab, scipy

