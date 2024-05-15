#!/usr/bin/python

from dgt import dgt

if __name__ == "__main__":

    ##############################################################
    ######## Check out the Model Parameter Space explorer ########
    ########    http://www.densegastoolbox.com/explorer   ########
    ##############################################################

    ##############################################################
    #################### USER INPUT BELOW ########################
    ##############################################################
    obsdata_file = 'ascii_galaxy.txt'    # table of observed intensities in [K km/s]
    
    ###################################
    # Note that the input file (obsdata_file) must have a 1-line
    # header, indicating the line intensities (in K km/s) via the
    # following column names:
    # 
    # CO10      CO21        CO32
    # HCN10     HCN21       HCN32
    # HCOP10    HCOP21      HCOP32
    # HNC10     HNC21       HNC32
    # 13CO10    13CO21      13CO32
    # C18O10    C18O21      C18O32
    # C17O10    C17O21      C17O32
    # CS10      CS21        CS32
    #
    # The uncertainties are similar, but starting with "UC_",
    # e.g. UC_CO10 or UC_HCN21

    ##############################################################
    # User Parameters
    powerlaw=False                           # logNorm or logNorm+PL density distribution
    T=0                                     # gas temperature; use T=0 to leave as free parameter
                                            # must be one of: 10,15,20,25,30,35,40,45,50
    W=0                                     # with of density distribution in dex; use W=0 to leave as free parameter
                                            # must be one of: 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    snr_line='13CO10'                       # only use data above SNR cut in given line, should be faintest line
    snr_lim=5.0                             # this is the corresponding SNR cut
    plotting=True                           # create plots
    domcmc=True                             # use MCMC for parameter estimation; this is recommended, but may take very long
    nsims=100                               # number of MCMC simulations to perform (should be >100 at least, better use 500+)
    ##############################################################

    # call Dense GasTool box
    dgt(obsdata_file,powerlaw,T,W,snr_line,snr_lim,plotting,domcmc,nsims)

    # exit
    exit(0)
