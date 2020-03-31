#!/usr/bin/python2.7

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
import re
from mpl_toolkits import mplot3d
from matplotlib import rc
from scipy.interpolate import Rbf
from scipy.stats import chi2 as scipychi2
from pylab import *
from read_grid_ndist import read_grid_ndist

import warnings

cmap='cubehelix'

# ignore some warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")

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

def scalar(array):
    if array.size==0:
        return -9.999999
    elif array.size==1:
        return np.asscalar(array)
    else:
        return np.asscalar(array[0])

##################################################################

def read_obs(filename):
    obsdata={}

    # read first line, used as dict keys
    with open(filename) as f:
        alllines=f.readlines()
        line=alllines[0]

        # read keys
        keys=re.sub('\s+',' ',line).strip().split(' ')

    f.close()

    # read values/columns
    with open(filename) as f:
        alllines=f.readlines()
        lines=alllines[1:]
        for i in range(len(keys)):
            get_col = lambda col: (re.sub('\s+',' ',line).strip().split(' ')[i] for line in lines)
            val=np.array(map(float,get_col(i)),dtype=np.float64)
            obsdata[keys[i]]=val
            keys[i] + ": "+str(val) 
    f.close()

    return obsdata

##################################################################

def write_result(result,outfile):
    result=np.array(result)

    # extract the results
    r=result.transpose()
    ra,de,chi2,n,T,width,densefrac=r[0],r[1],r[2],r[3],r[4],r[5],r[6]

    # save into new file
    out=np.column_stack((ra,de,chi2,n,T,width,densefrac))
    np.savetxt(outfile,out,\
      fmt="%.8f\t%.8f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f", \
      header="RA\tDEC\tchi2\tn\tT\twidth\tdensefrac")

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

        fig.savefig(pngoutfile,fig_layout='tight')
        plt.close()




##################################################################
##################################################################
##################################################################
################# The Dense Gas Toolbox ##########################
##################################################################
##################################################################

def dgt(obsdata_file,powerlaw,userT,userWidth,snr_line,snr_lim,plotting):

    # check user inputs (T and width)
    valid_T=[0,10,15,20,25,30,35,40,45,50]
    valid_W=[0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    if userT in valid_T and userWidth in valid_W:
        userinputOK=True
    else:
        userinputOK=False
        print "!!! User input (temperature or width) invalid. Exiting."
        print "!!!"
        exit()

    # Valid (i.e. modeled) input molecular lines are:

    valid_lines=['CO10','CO21','CO32',\
        'HCN10','HCN21','HCN32',\
        'HCOP10','HCOP21','HCOP32',\
        'HNC10','HNC21','HNC32',\
        '13CO10','13CO21','13CO32',\
        'C18O10','C18O21','C18O32',\
        'C17O10','C17O21','C17O32'\
    ]

    ###########################
    ### get observations ######
    ###########################

    obs=read_obs(obsdata_file)


    ###########################
    ##### validate input ######
    ###########################

    # check for coordinates in input file
    if not 'RA' in obs.keys() or not 'DEC' in obs.keys():
        print "!!!"
        print "!!! No coordinates found in input ascii file. Check column header for 'RA' and 'DEC'. Exiting."
        print "!!!"
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
        print "!!!"
        print "!!! Number of observed lines too low. Degrees of Freedom <1. Try a fixed temperature or check column header. Valid lines are: "
        print valid_lines
        print "!!!"
        exit()

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
        print "No CO line found in input data file. Check column headers for 'CO10', 'CO21' or 'CO32'. Exiting."
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
        if t<>normtrans:
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
            if t<>normtrans:
                uc='UC_'+t
                if obs[t][p]>obs[uc][p] and obs[t][p]>0.0:
                    diff[t]=np.array(((lr[t][p]-mdl[t])/lr[uc][p])**2)
                else:
                    diff[t]=np.nan*np.zeros_like(mdl[t])

        # vertical stack of diff arrays
        vstack=np.vstack(diff.values())
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
        n=ma.array(grid_n)

        grid_T=mdl['T']
        T=ma.array(grid_T)



        ###########################################################
        ########## find best fit set of parameters ################
        ################### from chi2 credible interval ###########
        ###########################################################

        # These limits correspond to +/-1 sigma error
        if dgf>0:
            #cutoff=0.05  # area to the right of critical value; here 5% --> 95% confidence  --> +/- 2sigma
            cutoff=0.32  # area to the right of critical value; here 32% --> 68% confidence --> +/- 1sigma
            deltachi2=scipychi2.ppf(1-cutoff, dgf)
        else:
            print "DGF is zero or negative."


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

        # the credible interval --> use for error estimate
        # not yet implemented though --> will be done in later version
        index_credible_int=ma.where(chi2<chi2+deltachi2)

        if len(index_credible_int[0])>3:

            quantiles=[0.0,1.0]

            # parameter limits/error within interval
            interval=T[index_credible_int]
            e1_bestT,e2_bestT=np.quantile(interval,quantiles)

            interval=n[index_credible_int]
            e1_bestn,e2_bestn=np.quantile(interval,quantiles)

            interval=width[index_credible_int]
            e1_bestwidth,e2_bestwidth=np.quantile(interval,quantiles)

            interval=densefrac[index_credible_int]
            e1_bestdensefrac,e2_bestdensefrac=np.quantile(interval,quantiles)

            result.append([ra[p],de[p],bestchi2,bestn,bestT,bestwidth,bestdensefrac])

        """
        else:
            result.append([ra[p],de[p],-99999,-99999,-99999,-99999,-99999])
        """


        ############################################
        ########## Plot result on screen ###########
        ############################################

        if SNR>snr_lim and bestn>0:
            print "\n#### Bestfit Parameters for pixel nr. "+str(p+1)+" ("+str(round(ra[p],5))+","+str(round(de[p],5))+ ") ####"
            print "chi2\t\t" + str(bestchi2)
            print "reduced chi2\t\t" + str(bestreducedchi2)
            print "n\t\t" + str(bestn)
            print "T\t\t" + str(bestT)
            print
            print "Width\t\t" + str(bestwidth)
            print "densefrac\t\t" + str(bestdensefrac)
            print




        ############################################
        ################ Make Figures ##############
        ############################################

        # Plotting
        if SNR>snr_lim and plotting==True and bestn>0:

            #### Create directory for output png files ###
            if not os.path.exists('./results/'):
                os.makedirs('./results/')

            # zoom-in variables
            zoom_n=n[chi2<bestchi2+deltachi2].compressed()
            zoom_chi2=chi2[chi2<bestchi2+deltachi2].compressed()
            zoom_width=width[chi2<bestchi2+deltachi2].compressed()


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
    outtable=obsdata_file[:-4]+"_nT.txt"
    resultfile="./results/"+outtable
    write_result(result,resultfile)

