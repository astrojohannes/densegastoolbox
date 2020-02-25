#!/usr/bin/python2.7

from dgt import dgt

if __name__ == "__main__":

    ##############################################################
    #################### USER INPUT BELOW ########################
    ##############################################################
    obsdata_file = 'ascii_galaxy.txt'       # table of observed intensities in [K km/s]
    powerlaw=True                           # logNorm or logNorm+PL density distribution
    T=0                                     # gas temperature; use T=0 to leave as free parameter
    snr_line='HCN10'                        # only use data above SNR cut in given line
    snr_lim=20.0                            # this is the corresponding SNR cut
    plotting=True                           # create plots
    ##############################################################

    # call Dense GasTool box
    dgt(obsdata_file,powerlaw,T,snr_line,snr_lim,plotting)

    # exit
    exit(0)
