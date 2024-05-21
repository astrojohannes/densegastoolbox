#!/usr/bin/env python

######################################################################################
# This takes observed intensities I of multiple lines from a table, e.g.
# 'ascii_galaxy.txt' and uses line ratios to perform a chi2 test on a
# radiative transfer model grid. The relative abundances (to CO) are fixed.
# The main result is a new table, e.g. 'ascii_galaxy_nT.txt'.
######################################################################################

import os
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
from matplotlib import rc
from scipy.interpolate import Rbf, LinearNDInterpolator
from scipy.stats import chi2 as scipychi2
from pylab import *
from read_grid_ndist import read_grid_ndist
import emcee
from multiprocessing import Pool
from datetime import datetime
import warnings
from mcmc_corner_plot import mcmc_corner_plot

cmap='cubehelix'

DEBUG=False

# ignore some warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="overflow encountered in power")


##################################################################

mpl.rc('lines', linewidth=3)
mpl.rc('axes', linewidth=2)
mpl.rc('xtick.major', size=4)
mpl.rc('ytick.major', size=4)
mpl.rc('xtick.minor', size=2)
mpl.rc('ytick.minor', size=2)
mpl.rc('axes', grid=False)
mpl.rc('xtick.major', width=1)
mpl.rc('xtick.minor', width=1)
mpl.rc('ytick.major', width=1)
mpl.rc('ytick.minor', width=1)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

##################################################################

def mymcmc(grid_theta, grid_loglike, ndim, nwalkers, backend, interp, nsims):

    ##### Define parameter grid for random selection of initial points for walker #######
    ##### PARAMETER GRID #####
    grid_n=10.**(1.8+np.arange(33)*0.1)
    grid_T=[10,15,20,25,30,35,40,45,50]
    grid_width=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    pos = [np.array([ \
           np.random.choice(grid_n,size=1)[0],\
           np.random.choice(grid_T,size=1)[0],\
           np.random.choice(grid_width,size=1)[0]]\
           ,dtype=np.float64) for i in range(nwalkers)]

    # theta=[n,T,width]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, getloglike, args=([grid_theta, grid_loglike, interp]), pool=pool, backend=backend)
        sampler.run_mcmc(pos, nsims, progress=True, store=True)

############################################################

def getloglike_nearest(theta, grid_theta, grid_loglike):

    intheta=np.array(theta,dtype=np.float64)

    diff=np.ones_like(grid_loglike)*1e10
    isclose=np.zeros_like(grid_loglike,dtype=np.bool)

    for i in range(len(grid_theta.T)):
        # calculate element-wise quadratic difference and sum it up
        # to get index of nearest neighbour on grid     
        diff[i]=((intheta-grid_theta.T[i])**2.0).sum()
        isclose[i]=np.allclose(intheta,grid_theta.T[i],rtol=0.50)

    # find index of nearest neighbour
    ind=np.array(diff,dtype=np.float64).argmin()

    """
    # sanity check: compare the found neighbor to the input parameters
    print(intheta,grid_theta.T[ind],grid_loglike[ind])
    """

    # check if the nearest neighbour is within 50% tolerance
    if not isclose[ind]:
        return -np.inf

    # check if the result is unambiguous (i.e. more than one minimum)
    if isinstance(ind,(list,np.ndarray)):
        print("WARNING Unambiguous grid point.")
        print(intheta)
        return -np.inf

    if not np.isfinite(grid_loglike[ind]):
        return -np.inf

    return grid_loglike[ind]

#####################################################################

def getloglike(theta, grid_theta, grid_loglike, interp):

    intheta=np.array(theta,dtype=np.float64)

    ###########################
    # nearest neighbour loglike
    if not interp:
        diff=np.ones_like(grid_loglike)*1e10
        isclose = np.zeros_like(grid_loglike, dtype=bool)

        for i in range(len(grid_theta.T)):
            # calculate element-wise quadratic difference and sum it up
            # to get index of nearest neighbour on grid     
            diff[i]=((intheta-grid_theta.T[i])**2.0).sum()
            isclose[i]=np.allclose(intheta,grid_theta.T[i],rtol=0.50)

        # find nearest neighbour
        ind=np.array(diff,dtype=np.float64).argmin()
        this_loglike=grid_loglike[ind]

        # check if the nearest neighbour is within 50% tolerance
        if not isclose[ind]:
            return -np.inf

        if not np.isfinite(this_loglike):
            return -np.inf


    #############################
    
    #############################
    # interpolated loglike
    else:
        interpol_func = LinearNDInterpolator(grid_theta.T, grid_loglike, fill_value=-np.inf, rescale=False)
        this_loglike=interpol_func(intheta)

        if not np.isfinite(this_loglike):
            return -np.inf

        this_loglike=float(this_loglike)

    #############################

    #print(intheta,grid_loglike[ind],this_loglike)

    return this_loglike


#####################################################################

def scalar(array):
    if array.size==0:
        return -9.999999
    elif array.size==1:
        return array.item()
    else:
        return array[0].item()

##################################################################

def read_obs(filename):
    obsdata={}

    # read first line, used as dict keys
    with open(filename) as f:
        alllines=f.readlines()
        line=alllines[0].replace('#','').replace('# ','').replace('#\t','')

        # read keys
        keys=re.sub('\s+',' ',line).strip().split(' ')

    f.close()

    # read values/columns
    with open(filename) as f:
        alllines=f.readlines()
        lines=alllines[1:]
        for i in range(len(keys)):
            get_col = lambda col: (re.sub('\s+',' ',line).strip().split(' ')[i] for line in lines if line)
            val=np.array([float(a) for a in get_col(i)],dtype=np.float64)
            obsdata[keys[i]]=val
            keys[i] + ": "+str(val) 
    f.close()

    return obsdata

##################################################################

def write_result(result,outfile,domcmc):
    result=np.array(result,dtype=object)

    tmpoutfile=outfile+'.tmp'

    # extract the results
    r=result.transpose()

    if not domcmc:
        ra,de,cnt,dgf,chi2,n,T,width,str_lines=r
        out=np.column_stack((ra,de,cnt,dgf,chi2,n,T,width,str_lines))
        np.savetxt(tmpoutfile,out,\
            fmt="%.8f\t%.8f\t%d\t%d\t%.4f\t%.2f\t%.2f\t%.2f\t%s", \
            header="RA\tDEC\tcnt\tdgf\tchi2\tn\tT\twidth\tlines_obs")
    else:
        ra,de,cnt,dgf,n,n_up,n_lo,T,T_up,T_lo,width,width_up,width_lo,str_lines=r
        out=np.column_stack((ra,de,cnt,dgf,n,n_up,n_lo,T,T_up,T_lo,width,width_up,width_lo,str_lines))
        np.savetxt(tmpoutfile,out,\
            fmt="%.8f\t%.8f\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s", \
            header="RA\tDEC\tcnt\tdgf\tn\te_n1\te_n2\tT\te_T1\te_T2\twidth\te_width1\te_width2\tlines_obs")

    # clean up
    replacecmd="sed -e\"s/', '/|/g;s/'//g;s/\[//g;s/\]//g\""
    os.system("cat "+tmpoutfile + "| "+ replacecmd + " > " + outfile)
    os.system("rm -rf "+tmpoutfile)

    return 

##################################################################

def makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile):
        fig = plt.figure(figsize=(7.5,6))
        ax = plt.gca()
        sliceindexes=np.where(this_slice==this_bestval)
        slicex=x[sliceindexes]
        slicey=y[sliceindexes]
        slicez=z[sliceindexes]
        slicex=np.array(slicex)
        slicey=np.array(slicey)
        slicez=np.array(slicez)

        if len(slicez)>3:
            # Set up a regular grid of interpolation points
            xi, yi = np.linspace(slicex.min(), slicex.max(), 60), np.linspace(slicey.min(), slicey.max(), 60)
            xi, yi = np.meshgrid(xi, yi)
            # Interpolate using Rbf
            rbf = Rbf(slicex, slicey, slicez, function='cubic')
            zi = rbf(xi, yi)

            q=[0.999]
            vmax=np.quantile(slicez,q)
            zi[zi>vmax]=vmax

            # replace nan with vmax (using workaround)
            val=-99999.9
            zi[zi==0.0]=val
            zi=np.nan_to_num(zi)
            zi[zi==0]=vmax
            zi[zi==val]=0.0

            # plot
            pl2=plt.imshow(zi, vmin=slicez.min(), vmax=slicez.max(), origin='lower', extent=[slicex.min(), slicex.max(), slicey.min(), slicey.max()],aspect='auto',cmap=cmap)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            clb=fig.colorbar(pl2)
            clb.set_label(label=zlabel,size=16)
            clb.ax.tick_params(labelsize=18)
        #####################################

        fig.subplots_adjust(left=0.13, bottom=0.12, right=0.93, top=0.94, wspace=0, hspace=0)
        fig = gcf()

        fig.suptitle(title, fontsize=18, y=0.99)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        fig.savefig(pngoutfile,bbox_inches='tight')
        plt.close()

        ######################################


##################################################################
##################################################################
##################################################################
################# The Dense Gas Toolbox ##########################
##################################################################
##################################################################

def dgt(obsdata_file,powerlaw,userT,userWidth,snr_line,snr_lim,plotting,domcmc,nsims):

    interp=False    # interpolate loglike on model grid (for mcmc sampler)
                    # this is not used yet, because needs some fixing

    # check user inputs (T and width)
    valid_T=[0,10,15,20,25,30,35,40,45,50]
    valid_W=[0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    if userT in valid_T and userWidth in valid_W:
        userinputOK=True
    else:
        userinputOK=False
        print("!!! User input (temperature or width) invalid. Exiting.")
        print("!!!")
        exit()

    # Valid (i.e. modeled) input molecular lines are:

    valid_lines=['CO10','CO21','CO32',\
        'HCN10','HCN21','HCN32',\
        'HCOP10','HCOP21','HCOP32',\
        'HNC10','HNC21','HNC32',\
        '13CO10','13CO21','13CO32',\
        'C18O10','C18O21','C18O32',\
        'C17O10','C17O21','C17O32',\
        'CS10','CS21','CS32'\
    ]

    if not snr_line in valid_lines:
        print("!!! Line for SNR limit is invalid. Must be one of:")
        print(valid_lines)
        print("!!!")
        exit()

    ###########################
    ### get observations ######
    ###########################

    obs=read_obs(obsdata_file)


    ###########################
    ##### validate input ######
    ###########################

    # check for coordinates in input file
    have_radec=False
    have_ra_special=False

    if 'RA' in obs.keys() and 'DEC' in obs.keys():
        have_radec=True
    elif '#RA' in obs.keys() and 'DEC' in obs.keys():
        have_radec=True
        have_ra_special=True
    else:
        have_radec=False


    if not have_radec:
        print("!!!")
        print("!!! No coordinates found in input ascii file. Check column header for 'RA' and 'DEC'. Exiting.")
        print("!!!")
        exit()

        
    # count number of lines in input data
    ct_l=0
    obstrans=[]    # this list holds the user input line keys
    for key in obs.keys():
        if key in valid_lines:
            ct_l+=1
            obstrans.append(key)

    # Only continue if number of molecular lines is > number of free parameters:
    if userT>0 and userWidth>0: dgf=ct_l-1
    elif userT>0 or userT>0: dgf=ct_l-2      # degrees of freedom = nrlines-2 if temperature is fixed. Free parameters: n,width
    else: dgf=ct_l-3        # Free parameters: n,T,width

    if not dgf>0:
        print("!!!")
        print("!!! Number of observed lines too low. Degrees of Freedom <1. Try a fixed temperature or check column header. Valid lines are: ")
        print(valid_lines)
        print("!!!")
        exit()

    if have_ra_special:
        ra=np.array(obs['#RA'])
    else:
        ra=np.array(obs['RA'])

    de=np.array(obs['DEC'])


    #############################################################################
    # Check input observations for lowest J CO line (used for normalization)
    #############################################################################
    have_co10=False
    have_co21=False
    have_co32=False

    # loop through observed lines/transitions
    for t in obstrans:
        if t=='CO10': have_co10=True
        if t=='CO21': have_co21=True
        if t=='CO32': have_co32=True

    if have_co10: normtrans='CO10'; uc_normtrans='UC_CO10'
    elif have_co21: normtrans='CO21'; uc_normtrans='UC_CO21'
    elif have_co32: normtrans='CO32'; uc_normtrans='UC_CO32'
    else:
        print("No CO line found in input data file. Check column headers for 'CO10', 'CO21' or 'CO32'. Exiting.")
        exit()



    ###########################
    ##### get the models ######
    ###########################
    mdl={}
    mdl = read_grid_ndist(obstrans,userT,userWidth,powerlaw)



    #############################################################################
    #############################################################################
    # Calculate line ratios and save in new dictionary
    # use line ratios (normalize to lowest CO transition in array) to determine chi2
    # note that the abundances are fixed by design of the model grid files
    #############################################################################
    #############################################################################
    lr={}

    # loop through observed lines/transitions
    for t in obstrans:
        if t!=normtrans:
            # calc line ratios
            lr[t]=obs[t]/obs[normtrans]
            mdl[t]=mdl[t]/mdl[normtrans]

            uc='UC_'+t
            lr[uc]=abs(obs[uc]/obs[t]) + abs(obs[uc_normtrans]/obs[normtrans])



    #############################################################
    #############################################################
    # loop through pixels, i.e. rows in ascii input file
    #############################################################
    #############################################################
    result=[]
    for p in range(len(ra)):

        #################################
        ####### calculate chi2 ##########
        #################################
        diff={}
        for t in obstrans:
            if t!=normtrans:
                uc='UC_'+t
                if obs[t][p]>obs[uc][p] and obs[t][p]>0.0:
                    diff[t]=np.array(((lr[t][p]-mdl[t])/lr[uc][p])**2)
                else:
                    diff[t]=np.nan*np.zeros_like(mdl[t])

        # vertical stack of diff arrays
        vstack=np.vstack(list(diff.values()))
        # sum up diff of all line ratios--> chi2
        chi2=vstack.sum(axis=0)

        # if model correct, we expect:
        # nu^2 ~ nu +/- sqrt(2*nu)

        # make a SNR cut using line and limit from user
        uc='UC_'+snr_line
        SNR=round(obs[snr_line][p]/obs[uc][p],2)

        width=ma.array(mdl['width'])
        densefrac=ma.array(mdl['densefrac'])

        # filter out outliers
        chi2lowlim,chi2uplim=np.quantile(chi2,[0.0,0.95])

        # create masks
        # invalid (nan) values of chi2
        chi2=ma.masked_invalid(chi2)
        mchi2invalid=ma.getmask(chi2)

        # based on chi2
        chi2=ma.array(chi2)
        chi2=ma.masked_outside(chi2, chi2lowlim, chi2uplim)
        mchi2=ma.getmask(chi2)
        # based on densefrac
        densefraclowlim=0.
        densefracuplim=99999. 
        densefrac=ma.masked_outside(densefrac,densefraclowlim,densefracuplim)
        mwidth=ma.getmask(densefrac)
 
        # combine masks
        m1=ma.mask_or(mchi2,mwidth)
        m=ma.mask_or(m1,mchi2invalid)

        width=ma.array(width,mask=m) 
        densefrac=ma.array(densefrac,mask=m)
        chi2=ma.array(chi2,mask=m)

        # n,T
        grid_n=mdl['n']
        n=ma.array(grid_n,mask=m)

        grid_T=mdl['T']
        T=ma.array(grid_T,mask=m)



        ###########################################################
        ########## find best fit set of parameters ################
        ################### from chi2 credible interval ###########
        ###########################################################

        # These limits correspond to +/-1 sigma error
        if dgf>0:
            cutoff=0.05  # area to the right of critical value; here 5% --> 95% confidence  --> +/- 2sigma
            #cutoff=0.32  # area to the right of critical value; here 32% --> 68% confidence --> +/- 1sigma
            deltachi2=scipychi2.ppf(1-cutoff, dgf)
        else:
            print("DGF is zero or negative.")

        # The minimum
        # find best fit set of parameters 
        chi2min=np.ma.min(chi2)
        bestfitindex=ma.where(chi2==chi2min)[0]
        bestchi2=scalar(chi2[bestfitindex].data)
        bestn=scalar(n[bestfitindex].data)
        bestwidth=scalar(width[bestfitindex].data)
        bestT=scalar(T[bestfitindex].data)
        bestdensefrac=scalar(densefrac[bestfitindex].data)
        bestchi2=round(bestchi2,2)
        bestreducedchi2=round(bestchi2/dgf,2)

        #################################################
        ########## Show Chi2 result on screen ###########
        #################################################

        if not domcmc:
            if SNR>snr_lim and bestn>0:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("#### Bestfit Parameters for pixel nr. "+str(p+1)+" ("+str(round(ra[p],5))+","+str(round(de[p],5))+ ") ####")
                print("chi2\t\t" + str(bestchi2))
                print("red. chi2\t\t" + str(bestreducedchi2))
                print("n\t\t" + str(bestn))
                print("T\t\t" + str(bestT))
                print("Width\t\t" + str(bestwidth))
                print()

                #############################################
                # save results in array for later file export
                result.append([ra[p],de[p],ct_l,dgf,bestchi2,bestn,bestT,bestwidth,obstrans])
                do_this_plot=True
            else:
                print("!-!-!-!-!-!")
                print("Pixel no. " +str(p+1)+ " --> SNR too low or density<0.")
                print()
                result.append([ra[p],de[p],ct_l,dgf,-99999.9,-99999.9,-99999.9,-99999.9,obstrans])
                do_this_plot=False

        ###################################################################
        ###################################################################
        ################################# MCMC ############################
        ###################################################################

        if domcmc:
            if SNR>snr_lim and bestn>0:

                #### Create directory for output png files ###
                if not os.path.exists('./results/'):
                    os.makedirs('./results/')

                starttime=datetime.now()

                ndim, nwalkers = 3, 50

                # model grid in results file
                grid_theta = np.array([n,T,width],dtype=np.float64)
                grid_loglike  = -0.5 * 10**chi2     # note that variable "chi2" is in fact log10(chi2) here

                if DEBUG:
                    from tabulate import tabulate
                    print("LOGLIKE")
                    print(tabulate(pd.DataFrame(grid_loglike), headers='keys', tablefmt='psql'))
                    print("88888888888888888888888888888888888888888888888888888888888")
                    print()


                # Set up the backend
                # Don't forget to clear it in case the file already exists
                status_filename = "./results/"+obsdata_file[:-4]+"_mcmc_"+str(p+1)+".h5"

                backend = emcee.backends.HDFBackend(status_filename)
                backend.reset(nwalkers, ndim)

                #### main ####
                mymcmc(grid_theta, grid_loglike, ndim, nwalkers, backend, interp, nsims)
                ##############

                duration=datetime.now()-starttime
                print("Duration for Pixel "+str(p+1)+": "+str(duration.seconds)+"sec")

                ########## MAKE CORNER PLOT #########
                outpngfile="./results/"+obsdata_file[:-4]+"_mcmc_"+str(p+1)+".png"
                bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower=mcmc_corner_plot(status_filename,outpngfile)

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("#### Bestfit Parameters for pixel nr. "+str(p+1)+" ("+str(round(ra[p],5))+","+str(round(de[p],5))+ ") ####")
                print("n\t\t" + str(bestn_mcmc_val) + " " + str(bestn_mcmc_upper) + " " + str(bestn_mcmc_lower))
                print("T\t\t" + str(bestT_mcmc_val) + " " + str(bestT_mcmc_upper) + " " + str(bestT_mcmc_lower))
                print("Width\t\t" + str(bestW_mcmc_val) + " " + str(bestW_mcmc_upper) + " " + str(bestW_mcmc_lower))
                print()


                #############################################
                # save results in array for later file export
                result.append([ra[p],de[p],ct_l,dgf,float(bestn_mcmc_val),float(bestn_mcmc_upper),float(bestn_mcmc_lower),\
                                                    float(bestT_mcmc_val),float(bestT_mcmc_upper),float(bestT_mcmc_lower),\
                                                    float(bestW_mcmc_val),float(bestW_mcmc_upper),float(bestW_mcmc_lower),obstrans])
                do_this_plot=True 
                ###################################################################
                ###################################################################


            else:
                do_this_plot=False

        ############################################
        ################ Make Figures ##############
        ############################################

        # Plotting
        if SNR>snr_lim and plotting==True and bestn>0 and do_this_plot:

            #### Create directory for output png files ###
            if not os.path.exists('./results/'):
                os.makedirs('./results/')

            # zoom-in variables
            idx=np.where(chi2<bestchi2+deltachi2)
            zoom_n=n[idx].compressed()
            zoom_chi2=chi2[idx].compressed()
            zoom_width=width[idx].compressed()

            ########################## PLOT 1 #############################

            # combine 4 plots to a single file
            fig, ax = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(11.5,8))
            # Chi2 vs n plot

            ax[0,0].scatter(chi2, np.log10(n),c=width, cmap='Accent',marker=',',s=4,vmin=width.min(),vmax=width.max())
            ax[0,0].set_ylabel('$log\ n$') 

            pl1=ax[0,1].scatter(zoom_chi2, np.log10(zoom_n),c=zoom_width, cmap='Accent',marker=',',s=9,vmin=width.min(),vmax=width.max())
            fig.colorbar(pl1,ax=ax[0,1],label='$\mathsf{width}$')

            # Chi2 vs T plot
            ax[1,0].scatter(chi2, np.log10(T),c=width, cmap='Accent',marker=',',s=4,vmin=width.min(),vmax=width.max())
            ax[1,0].set_xlabel('$\chi^2$')
            ax[1,0].set_ylabel('$log\ T$') 

            # Chi2 vs T plot zoom-in
            zoom_T=T[chi2<bestchi2+deltachi2].compressed()
            pl2=ax[1,1].scatter(zoom_chi2, np.log10(zoom_T),c=zoom_width, cmap='Accent',marker=',',s=9,vmin=width.min(),vmax=width.max())
            ax[1,1].set_xlabel('$\chi^2}$')
            fig.colorbar(pl2,ax=ax[1,1],label='$\mathsf{width}$')
 
            # plot
            fig.subplots_adjust(left=0.06, bottom=0.06, right=1, top=0.96, wspace=0.04, hspace=0.04)
            fig = gcf()
            fig.suptitle('Pixel: ('+str(p)+') SNR('+snr_line+'): '+str(SNR), fontsize=14, y=0.99) 
            chi2_filename=obsdata_file[:-4]+"_"+str(p+1)+'_chi2.png'
            fig.savefig('./results/'+chi2_filename) 
            #plt.show()
            plt.close()


            ########################## PLOT 2 #############################
            # all parameters free: (n,T) vs. chi2
            if userT==0 and userWidth==0:

                  x=np.log10(zoom_n)
                  y=np.log10(zoom_T)
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_width
                  this_bestval=bestwidth
                  xlabel='$log\ n\ [cm^{-3}]$'
                  ylabel='$log\ T\ [K]$'
                  zlabel='$\mathsf{log\ \chi^2}$'

                  title='Pixel: '+str(p+1)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results/'+obsdata_file[:-4]+"_"+str(p+1)+'_nT.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)

                  ########################## PLOT 3 #############################
                  # all parameters free: (n,width) vs. chi2
                  x=np.log10(zoom_n)
                  y=zoom_width
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_T
                  this_bestval=bestT
                  xlabel='$log\ n\ [cm^{-3}]$'
                  ylabel='$width\ [dex]$'
                  zlabel='$\mathsf{log\ \chi^2}$'

                  title='Pixel: '+str(p+1)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results/'+obsdata_file[:-4]+"_"+str(p+1)+'_nW.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)
 
            # width fixed: (n,T) vs. chi2
            elif userT==0 and userWidth>0:
                  x=np.log10(zoom_n)
                  y=np.log10(zoom_T)
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_width
                  this_bestval=bestwidth
                  xlabel='$log\ n\ [cm^{-3}]$'
                  ylabel='$log\ T\ [K]$'
                  zlabel='$\mathsf{log\ \chi^2}$'

                  title='Pixel: '+str(p+1)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results/'+obsdata_file[:-4]+"_"+str(p+1)+'_nT_fixedW.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)

            # T fixed: (n,width) vs. chi2
            elif userT>0 and userWidth==0:
                  x=np.log10(zoom_n)
                  y=zoom_width
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_T
                  this_bestval=bestT
                  xlabel='$log\ n\ [cm^{-3}]$'
                  ylabel='$width\ [dex]$'
                  zlabel='$\mathsf{log\ \chi^2}$'

                  title='Pixel: '+str(p+1)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results/'+obsdata_file[:-4]+"_"+str(p+1)+'_nW_fixedT.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)


        del diff,chi2,n,T,width,densefrac,mchi2,mchi2invalid,mwidth,m1,m,grid_n,grid_T


    ################################################
    ################################################
    # write result to a new output table
    if not domcmc:
        outtable=obsdata_file[:-4]+"_nT.txt"
    else:
        outtable=obsdata_file[:-4]+"_nT_mcmc.txt"
    resultfile="./results/"+outtable
    write_result(result,resultfile,domcmc)

