#!/usr/bin/python2.7

from dgt import dgt

if __name__ == "__main__":

    ##############################################################
    ######## Check out the Model Parameter Space explorer ########
    ########    http://www.densegastoolbox.com/explorer   ########
    ##############################################################

    ##############################################################
    #################### USER INPUT BELOW ########################
    ##############################################################
    obsdata_file = 'ascii_galaxy.txt'       # table of observed intensities in [K km/s]
    powerlaw=True                           # logNorm or logNorm+PL density distribution
    T=0                                     # gas temperature; use T=0 to leave as free parameter
                                            # must be one of: 10,15,20,25,30,35,40,45,50
    W=0                                     # with of density distribution in dex; use W=0 to leave as free parameter
                                            # must be one of: 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    snr_line='HCN10'                        # only use data above SNR cut in given line
    snr_lim=10.0                            # this is the corresponding SNR cut
    plotting=True                           # create plots
    ##############################################################

    # call Dense GasTool box
    dgt(obsdata_file,powerlaw,T,W,snr_line,snr_lim,plotting)

    # exit
    exit(0)
