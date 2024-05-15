#!/usr/bin/python2.7

import numpy as np

#Column 1: tkin [K]
#Column 2: mean of density distribution [cm^-3]
#Column 3: 1sigma width of density distribution [dex]
#Column 4: power law tail implemented?
#Column 5: fraction of mass in powerlaw tail.
#Column 6: median density by mass
#Column 7: fraction of mass above 1e4.5 cm^-3
#Column 8: co10 emissivity (actual value K km/s per cm^-2)
#Column 9: median density by flux for co10

def limitarr(arr,index):
    arr=np.array(arr)
    out=[None for i in range(arr.shape[0])]
    for i in range(arr.shape[0]):
        a=arr[i]
        out[i]=a[index]

    del arr, a
    return out
    
############################################################

def read_grid_ndist(transition,usertkin,userwidth,powerlaw):
    m={}

    if not powerlaw:
        lratfile='lrat_table.txt'
    else:
        lratfile='lrat_table_powerlaw.txt'

    gridfile='models/'+lratfile
    Tkin,n_mean,width,pl,plmass,n_mean_mass,densefrac, \
            ICO,n_mean_ICO,\
            ICO21_ICO,n_mean_ICO21,\
            ICO32_ICO,n_mean_ICO32,\
            I13CO10_ICO,n_mean_I13CO10,\
            I13CO21_ICO,n_mean_I13CO21,\
            I13CO32_ICO,n_mean_I13CO32,\
            IC18O10_ICO,n_mean_IC18O10,\
            IC18O21_ICO,n_mean_IC18O21,\
            IC18O32_ICO,n_mean_IC18O32,\
            IC17O10_ICO,n_mean_IC17O10,\
            IC17O21_ICO,n_mean_IC17O21,\
            IC17O32_ICO,n_mean_IC17O32,\
            IHCN10_ICO,n_mean_IHCN10,\
            IHCN21_ICO,n_mean_IHCN21,\
            IHCN32_ICO,n_mean_IHCN32,\
            IHNC10_ICO,n_mean_IHNC10,\
            IHNC21_ICO,n_mean_IHNC21,\
            IHNC32_ICO,n_mean_IHNC32,\
            IHCOP10_ICO,n_mean_IHCOP10,\
            IHCOP21_ICO,n_mean_IHCOP21,\
            IHCOP32_ICO,n_mean_IHCOP32,\
            ICS10_ICO,n_mean_ICS10,\
            ICS21_ICO,n_mean_ICS21,\
            ICS32_ICO,n_mean_ICS32 \
            = np.loadtxt(gridfile, skiprows=55,unpack=True)

    # from variables to list
    mygrid=[Tkin,n_mean,width,pl,plmass,n_mean_mass,densefrac, \
            ICO,n_mean_ICO,\
            ICO21_ICO,n_mean_ICO21,\
            ICO32_ICO,n_mean_ICO32,\
            I13CO10_ICO,n_mean_I13CO10,\
            I13CO21_ICO,n_mean_I13CO21,\
            I13CO32_ICO,n_mean_I13CO32,\
            IC18O10_ICO,n_mean_IC18O10,\
            IC18O21_ICO,n_mean_IC18O21,\
            IC18O32_ICO,n_mean_IC18O32,\
            IC17O10_ICO,n_mean_IC17O10,\
            IC17O21_ICO,n_mean_IC17O21,\
            IC17O32_ICO,n_mean_IC17O32,\
            IHCN10_ICO,n_mean_IHCN10,\
            IHCN21_ICO,n_mean_IHCN21,\
            IHCN32_ICO,n_mean_IHCN32,\
            IHNC10_ICO,n_mean_IHNC10,\
            IHNC21_ICO,n_mean_IHNC21,\
            IHNC32_ICO,n_mean_IHNC32,\
            IHCOP10_ICO,n_mean_IHCOP10,\
            IHCOP21_ICO,n_mean_IHCOP21,\
            IHCOP32_ICO,n_mean_IHCOP32,\
            ICS10_ICO,n_mean_ICS10,\
            ICS21_ICO,n_mean_ICS21,\
            ICS32_ICO,n_mean_ICS32]

    # limit to reasonable range (defined by one-zone grid and width of distribution)
    # compare to http://www.densegastoolbox.com/explorer/
    n_lolim=10**1.8
    n_uplim=10**5.0

    n_mean=mygrid[1]
    index=np.where(n_mean>n_lolim)[0]
    mygrid=limitarr(mygrid,index)

    n_mean=mygrid[1]
    index=np.where(n_mean<n_uplim)[0]
    mygrid=limitarr(mygrid,index)

    # limit to values at user temperature
    if usertkin>0:
        Tkin=mygrid[0]
        index=np.where(Tkin==usertkin)[0]
        mygrid=limitarr(mygrid,index)

    # limit to values at user temperature
    if userwidth>0:
        width=mygrid[2]
        index=np.where(width==userwidth)[0]
        mygrid=limitarr(mygrid,index)

    # back from list to variables
    Tkin,n_mean,width,pl,plmass,n_mean_mass,densefrac, \
            ICO,n_mean_ICO,\
            ICO21_ICO,n_mean_ICO21,\
            ICO32_ICO,n_mean_ICO32,\
            I13CO10_ICO,n_mean_I13CO10,\
            I13CO21_ICO,n_mean_I13CO21,\
            I13CO32_ICO,n_mean_I13CO32,\
            IC18O10_ICO,n_mean_IC18O10,\
            IC18O21_ICO,n_mean_IC18O21,\
            IC18O32_ICO,n_mean_IC18O32,\
            IC17O10_ICO,n_mean_IC17O10,\
            IC17O21_ICO,n_mean_IC17O21,\
            IC17O32_ICO,n_mean_IC17O32,\
            IHCN10_ICO,n_mean_IHCN10,\
            IHCN21_ICO,n_mean_IHCN21,\
            IHCN32_ICO,n_mean_IHCN32,\
            IHNC10_ICO,n_mean_IHNC10,\
            IHNC21_ICO,n_mean_IHNC21,\
            IHNC32_ICO,n_mean_IHNC32,\
            IHCOP10_ICO,n_mean_IHCOP10,\
            IHCOP21_ICO,n_mean_IHCOP21,\
            IHCOP32_ICO,n_mean_IHCOP32,\
            ICS10_ICO,n_mean_ICS10,\
            ICS21_ICO,n_mean_ICS21,\
            ICS32_ICO,n_mean_ICS32 = mygrid


    # match variables from above to dict keys --> should be improved later
    # i.e. populate model dictionary
    m['CO10']=ICO
    m['CO21']=ICO21_ICO*ICO
    m['CO32']=ICO32_ICO*ICO

    m['13CO10']=I13CO10_ICO*ICO
    m['13CO21']=I13CO21_ICO*ICO
    m['13CO32']=I13CO32_ICO*ICO

    m['C18O10']=IC18O10_ICO*ICO
    m['C18O21']=IC18O21_ICO*ICO
    m['C18O32']=IC18O32_ICO*ICO

    m['C17O10']=IC17O10_ICO*ICO
    m['C17O21']=IC17O21_ICO*ICO
    m['C17O32']=IC17O32_ICO*ICO

    m['HCN10']=IHCN10_ICO*ICO
    m['HCN21']=IHCN21_ICO*ICO
    m['HCN32']=IHCN32_ICO*ICO

    m['HNC10']=IHNC10_ICO*ICO
    m['HNC21']=IHNC21_ICO*ICO
    m['HNC32']=IHNC32_ICO*ICO

    m['HCOP10']=IHCOP10_ICO*ICO
    m['HCOP21']=IHCOP21_ICO*ICO
    m['HCOP32']=IHCOP32_ICO*ICO

    m['CS10']=ICS10_ICO*ICO
    m['CS21']=ICS21_ICO*ICO
    m['CS32']=ICS32_ICO*ICO

    m['T']=Tkin
    m['n']=n_mean_mass
    m['width']=width
    m['densefrac']=densefrac
    
    return m

