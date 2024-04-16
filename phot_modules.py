"""

    All the functions listed here requires the generation of the particle
    information file and getting the los metal density.

"""

import numpy as np
import pandas as pd

import sys
import os
from functools import partial
import schwimmbad

import synthobs
from synthobs.sed import models

import flare
import flare.filters
from flare.photom import lum_to_M, M_to_lum

import h5py


def get_data(ii, tag, inp = 'FLARES', data_folder = 'data', aperture = '30'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num

        sim = rF"./{data_folder}/FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"


    with h5py.File(sim, 'r') as hf:

        S_len   = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
    
        begin       = np.zeros(len(S_len), dtype = np.int64)
        end         = np.zeros(len(S_len), dtype = np.int64)
        begin[1:]   = np.cumsum(S_len)[:-1]
        end         = np.cumsum(S_len)

        DTM     = np.array(hf[tag+'/Galaxy'].get('DTM'), dtype = np.float64)
        S_mass  = np.array(hf[tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        S_Z     = np.array(hf[tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age   = np.array(hf[tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        S_los   = np.array(hf[tag+'/Particle'].get('S_los'), dtype = np.float64)

        S_ap = {}
        if aperture == 'halfmassradius':
            half_mass_rad = np.array(hf[tag+'/Galaxy'].get('HalfMassRad'), dtype = np.float64)
            gsize = half_mass_rad[:,4] * 4 * 1e3
            apertures = np.array([1,3,5,10,20,30,40,50,70,100])
            ap_use = np.abs(np.array([gsize - _ap for _ap in apertures])).argmin(axis=0)
            
            ap = {}
            for _ap in apertures:
                ap[_ap] = np.array(hf[tag+'/Particle/Apertures/Star'].get(F'{_ap}'), dtype = np.bool)

            for i, j in enumerate(begin):
                S_ap[i] = ap[apertures[ap_use[i]]][begin[i]:end[i]]

        else:
            ap = np.array(hf[tag+'/Particle/Apertures/Star'].get(F'{aperture}'), dtype = np.bool)
            for i, j in enumerate(begin):
                S_ap[i] = ap[begin[i]:end[i]]

    return S_mass, S_Z, S_age, S_los, S_len, begin, end, S_ap, DTM


def lum(sim, kappa, tag, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    S_mass, S_Z, S_age, S_los, S_len, begin, end, S_ap, DTM = get_data(sim, tag, inp, data_folder, aperture)

    if np.isscalar(filters):
        Lums = np.zeros(len(begin), dtype = np.float64)
    else:
        Lums = np.zeros((len(begin), len(filters)), dtype = np.float64)

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM  = ('Starburst_Calzetti2000', {''})
        model.dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM  = ('SMC_Pei92', {''})
        model.dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM  = ('MW_Pei92', {''})
        model.dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM  = ('MW_N18', {''})
        model.dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p','.'))

    # --- create rest-frame luminosities
    F = flare.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz


    for jj in range(len(begin)):

        Masses              = S_mass[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        Ages                = S_age[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        Metallicities       = S_Z[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]

        MetSurfaceDensities = DTM[jj] * MetSurfaceDensities

        if Type == 'Total':
            tauVs_ISM   = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        else:
            ValueError(F"Undefined Type {Type}")


        Lnu         = models.generate_Lnu(model = model, F = F, Masses = Masses, Ages = Ages, Metallicities = Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = fesc, log10t_BC = log10t_BC) # --- calculate rest-frame Luminosity. In units of erg/s/Hz

        Lums[jj]    = list(Lnu.values())

    return Lums



def flux(sim, kappa, tag, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300', filters = flare.filters.NIRCam_W, Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    S_mass, S_Z, S_age, S_los, S_len, begin, end, S_ap, DTM = get_data(sim, tag, inp, data_folder, aperture)

    if np.isscalar(filters):
        Fnus = np.zeros(len(begin), dtype = np.float64)
    else:
        Fnus = np.zeros((len(begin), len(filters)), dtype = np.float64)

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM  = ('Starburst_Calzetti2000', {''})
        model.dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM  = ('SMC_Pei92', {''})
        model.dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM  = ('MW_Pei92', {''})
        model.dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM  = ('MW_N18', {''})
        model.dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p','.'))
    F = flare.filters.add_filters(filters, new_lam = model.lam * (1. + z))

    cosmo = flare.default_cosmo()

    model.create_Fnu_grid(F, z, cosmo) # --- create new Fnu grid for each filter. In units of nJy/M_sol

    for jj in range(len(begin)):

        Masses              = S_mass[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        Ages                = S_age[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        Metallicities       = S_Z[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]][S_ap[jj]] # [S_ap[begin[jj]:end[jj]]]

        MetSurfaceDensities = DTM[jj] * MetSurfaceDensities

        if Type == 'Total':
            tauVs_ISM   = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        else:
            ValueError(F"Undefined Type {Type}")

        Fnu         = models.generate_Fnu(model = model, F = F, Masses = Masses, Ages = Ages, Metallicities = Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = fesc, log10t_BC = log10t_BC) # --- calculate rest-frame flux of each object in nJy

        Fnus[jj]    = list(Fnu.values())

    return Fnus


def get_lines(line, sim, kappa, tag, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300', LF = False, Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    S_mass, S_Z, S_age, S_los, S_len, begin, end, S_ap, DTM = get_data(sim, tag, inp, data_folder, aperture)

    # --- calculate intrinsic quantities
    if extinction == 'default':
        dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        dust_ISM  = ('Starburst_Calzetti2000', {''})
        dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        dust_ISM  = ('SMC_Pei92', {''})
        dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        dust_ISM  = ('MW_Pei92', {''})
        dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        dust_ISM  = ('MW_N18', {''})
        dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    lum = np.zeros(len(begin), dtype = np.float64)
    EW = np.zeros(len(begin), dtype = np.float64)

    # --- initialise model with SPS model and IMF. Set verbose = True to see a list of available lines.
    m = models.EmissionLines(F'BPASSv2.2.1.binary/{IMF}', dust_BC = dust_BC, dust_ISM = dust_ISM, verbose = False)

    for jj in range(len(begin)):

        Masses              = S_mass[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        Ages                = S_age[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        Metallicities       = S_Z[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]

        MetSurfaceDensities = DTM[jj] * MetSurfaceDensities

        if Type == 'Total':
            tauVs_ISM   = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = np.zeros(len(Masses))
            fesc        = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM   = np.zeros(len(Masses))
            tauVs_BC    = BC_fac * (Metallicities/0.01)
            fesc        = 0.0

        else:
            ValueError(F"Undefined Type {Type}")


        o = m.get_line_luminosity(line, Masses, Ages, Metallicities, tauVs_BC = tauVs_BC, tauVs_ISM = tauVs_ISM, verbose = False, log10t_BC = log10t_BC)

        lum[jj] = o['luminosity']
        EW[jj] = o['EW']

    return lum, EW


def get_SED(sim, kappa, tag, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    S_mass, S_Z, S_age, S_los, S_len, begin, end, S_ap, DTM = get_data(sim, tag, inp, data_folder, aperture)

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    # --- calculate intrinsic quantities
    if extinction == 'default':
        model.dust_ISM  = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC   = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM  = ('Starburst_Calzetti2000', {''})
        model.dust_BC   = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM  = ('SMC_Pei92', {''})
        model.dust_BC   = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM  = ('MW_Pei92', {''})
        model.dust_BC   = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM  = ('MW_N18', {''})
        model.dust_BC   = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    for jj in range(len(begin)):

        Masses              = S_mass[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        Ages                = S_age[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        Metallicities       = S_Z[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]][S_ap[begin[jj]:end[jj]]]

        MetSurfaceDensities = DTM[jj] * MetSurfaceDensities

        tauVs_ISM = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC = BC_fac * (Metallicities/0.01)

        o = models.generate_SED(model, Masses, Ages, Metallicities, tauVs_ISM = tauVs_ISM, tauVs_BC = tauVs_BC, fesc = 0.0)

        if jj == 0:
            lam = np.zeros((len(begin), len(o.lam)), dtype = np.float64)        #in Angstrom
            stellar = np.zeros((len(begin), len(o.lam)), dtype = np.float64)
            intrinsic = np.zeros((len(begin), len(o.lam)), dtype = np.float64)
            no_ism = np.zeros((len(begin), len(o.lam)), dtype = np.float64)
            total = np.zeros((len(begin), len(o.lam)), dtype = np.float64)

        lam[jj] = o.lam
        stellar[jj] = o.stellar.lnu
        intrinsic[jj] = o.intrinsic.lnu
        no_ism[jj] = o.BC.lnu
        total[jj] = o.total.lnu

    return lam, stellar, intrinsic, no_ism, total



def get_lum(sim, kappa, tag, BC_fac, IMF = 'Chabrier_300', bins = np.arange(-24, -16, 0.5), inp = 'FLARES', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    # try:
    Lums = lum(sim, kappa, tag, BC_fac = BC_fac, IMF=IMF, inp=inp, LF=LF, filters=filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

    # except Exception as e:
    #     Lums = np.ones(len(filters))*np.nan
    #     print (e)


    if LF:
        tmp, edges = np.histogram(lum_to_M(Lums), bins = bins)
        return tmp

    else:
        return Lums



def get_lum_all(kappa, tag, BC_fac, IMF = 'Chabrier_300', bins = np.arange(-24, -16, 0.5), inp = 'FLARES', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    print (f"Getting luminosities for tag {tag} with kappa = {kappa}")

    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0,len(weights))

        calc = partial(get_lum, kappa = kappa, tag = tag, BC_fac = BC_fac, IMF = IMF, bins = bins, inp = inp, LF = LF, filters = filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis = 0)
            out = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            for ii, sim in enumerate(sims):
                err+=np.square(np.sqrt(dat[ii])*weights[ii])
                out+=dat[ii]*weights[ii]

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:
        out = get_lum(00, kappa = kappa, tag = tag, BC_fac = BC_fac, IMF = IMF, bins = bins, inp = inp, LF = LF, filters = filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

        return out


def get_flux(sim, kappa, tag, BC_fac,  IMF = 'Chabrier_300', inp = 'FLARES', filters = flare.filters.NIRCam, Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    try:
        Fnus = flux(sim, kappa, tag, BC_fac = BC_fac, IMF=IMF, inp=inp, filters=filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

    except Exception as e:
        Fnus = np.ones(len(filters))*np.nan
        print (e)

    return Fnus

def get_flux_all(kappa, tag, BC_fac, IMF = 'Chabrier_300', inp = 'FLARES', filters = flare.filters.NIRCam, Type = 'Total', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='30'):

    print (f"Getting fluxes for tag {tag} with kappa = {kappa}")

    if inp == 'FLARES':

        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0,len(weights))

        calc = partial(get_flux, kappa = kappa, tag = tag, BC_fac = BC_fac, IMF = IMF, inp = inp, filters = filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

        pool = schwimmbad.MultiPool(processes=8)
        out = np.array(list(pool.map(calc, sims)))
        pool.close()

    else:

        out = get_flux(00, kappa = kappa, tag = tag, BC_fac = BC_fac, IMF = IMF, inp = inp, filters = filters, Type = Type, log10t_BC = log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

    return out
