#!/usr/bin/env python

import emcee
import numpy as np
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys

def mcmc_corner_plot(infile,outfile):

    reader = emcee.backends.HDFBackend(infile)

    tau = reader.get_autocorr_time(tol=0)

    if not pd.isna(np.nanmax(tau)) and not pd.isna(np.nanmin(tau)):
        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau))
    
        samples = reader.get_chain(flat=True, discard=burnin, thin=thin)
        logprob = reader.get_log_prob(flat=True, discard=burnin, thin=thin)

        logprob[logprob==-np.inf]=-1e40
        logprob[logprob==np.inf]=1e40

        # check for convergence
        logprob_min=np.nanmin(logprob)
        logprob_max=np.nanmax(logprob)
    
        if logprob_max>logprob_min:
            converged=True
        else:
            converged=False
            print("Warning: MCMC did not converge, you may try to increase the number of simulations (nsims)!")
            return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

        """
        print()
        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(samples.shape))
        print("flat log prob shape: {0}".format(logprob.shape))
        print()
        """

        #all_samples = np.concatenate((samples, logprob[:, None]), axis=1)

        all_samples = samples

        labels=['n','T','width']
        #labels += ["log prob"]
    
        # 0.16 and 0.84 percentiles correspond to +/- 1 sigma in a Gaussian
        q=[0.16, 0.5, 0.84]
        figure=corner.corner(all_samples, labels=labels,\
            range=[(10,1e5),(10,100),(0.0,1.4)],\
            quantiles=q,\
            plot_datapoints=False,\
            plot_contours=True,\
            plot_density=True,\
            fill_contours=True,\
            contour_kwargs={'cmap':'viridis','colors':None},\
            contourf_kwargs={'cmap':'viridis','colors':None},\
            show_titles=True, title_kwargs={"fontsize": 16},\
            label_kwargs={"fontsize": 16}
        )
    
        # calculate quantile-based 1-sigma error bars
        samples_n=samples[:,0]
        samples_T=samples[:,1]
        samples_W=samples[:,2]
        q=[0.16,0.5,0.84]
        lowern_mcmc,bestn_mcmc,uppern_mcmc=np.quantile(samples_n,q)
        lowerT_mcmc,bestT_mcmc,upperT_mcmc=np.quantile(samples_T,q)
        lowerW_mcmc,bestW_mcmc,upperW_mcmc=np.quantile(samples_W,q)
    
        bestn_mcmc_val=str(round(bestn_mcmc,2))
        bestn_mcmc_upper="+"+str(round(uppern_mcmc-bestn_mcmc,2))
        bestn_mcmc_lower="-"+str(round(bestn_mcmc-lowern_mcmc,2))
    
        bestT_mcmc_val=str(round(bestT_mcmc,2))
        bestT_mcmc_upper="+"+str(round(upperT_mcmc-bestT_mcmc,2))
        bestT_mcmc_lower="-"+str(round(bestT_mcmc-lowerT_mcmc,2))
    
        bestW_mcmc_val=str(round(bestW_mcmc,2))
        bestW_mcmc_upper="+"+str(round(upperW_mcmc-bestW_mcmc,2))
        bestW_mcmc_lower="-"+str(round(bestW_mcmc-lowerW_mcmc,2))
   
        """ 
        # Extract the axes
        #axes = np.array(figure.axes).reshape((4, 4))
        axes = np.array(figure.axes).reshape((3, 3))
    
        # set x axis range
        #for yi in range(4):
        for yi in range(3):
            # density: 10**1.8 --> 10**5
            ax = axes[yi, 0]
            if yi==0: ax.set_xlim(10.0,100000.0)   # density
            if yi==1: ax.set_xlim(10,100)   # temperature
            if yi==2: ax.set_xlim(0.2,2.0)  # width
        """

        figure.savefig(outfile,bbox_inches='tight')

    else:
        print("Warning: Tau is NaN, corner plot cannot be created!")
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

    return [bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower]
