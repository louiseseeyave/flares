"""
    Calculates the line-of-sight metal density for star and SMBH particles using gas particles within a given aperture. At the moment the default is 30 pkpc same as the original EAGLE prescription
"""

import timeit, sys
import numpy as np
from numba import jit, njit, float64, int32, prange
import h5py
from functools import partial
import schwimmbad
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u

import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)
norm = np.linalg.norm

@njit(float64[:](float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int32), parallel=True, nogil=True)
def get_S_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in Msun/Mpc^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n = len(s_cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis.
    xdir, ydir, zdir = 0, 1, 2
    for ii in prange(n):

        thisspos = s_cood[ii]
        ok = np.where(g_cood[:,zdir] > thisspos[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2


    return Z_los_SD


@njit(float64(float64[:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int32), parallel=True, nogil=True)
def get_BH_LOS(bh_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in Msun/Mpc^2) for bh particles inside the galaxy taking
    the z-axis as the los.
    Args:
        bh_cood (3d array): BH particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """

    #Fixing the observer direction as z-axis.
    xdir, ydir, zdir = 0, 1, 2

    if np.sum(bh_cood)>0:
        ok = np.where(g_cood[:,zdir] > bh_cood[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - bh_cood[xdir]
        y = thisgpos[:,ydir] - bh_cood[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2

    else:
        Z_los_SD = 0.

    return Z_los_SD

def get_data(ii, tag, inp = 'FLARES', data_folder='data/'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        sim_type = 'FLARES'


    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    print (filename)

    with h5py.File(filename, 'r') as hf:
        S_len = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)

        S_coords = np.array(hf[tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        G_mass = np.array(hf[tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml = np.array(hf[tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z = np.array(hf[tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)
        BH_coords = np.array(hf[tag+'/Galaxy'].get('BH_Coordinates'), dtype = np.float64)

        S_ap = np.array(hf[tag+'/Particle/Apertures'].get('Star'), dtype = np.bool)
        G_ap = np.array(hf[tag+'/Particle/Apertures'].get('Gas'), dtype = np.bool)


    return S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len, BH_coords, S_ap, G_ap


def get_len(Length):

    begin = np.zeros(len(Length), dtype = np.int64)
    end = np.zeros(len(Length), dtype = np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)

    return begin, end


def get_ZLOS(jj, req_coords, G_coords, G_mass, G_Z, G_sml, gbegin, gend, lkernel, kbins, G_ap, S_ap=[], sbegin=[], send=[], Stars=True):


    if Stars:
        this_coords = req_coords[sbegin[jj]:send[jj]][S_ap[sbegin[jj]:send[jj]]]
        req_func = partial(get_S_LOS)
    else:
        this_coords = req_coords[jj]
        req_func = partial(get_BH_LOS)


    this_gcoords = G_coords[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gmass = G_mass[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gsml = G_sml[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]


    Z_los_SD = req_func(this_coords, this_gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins)*conv

    return Z_los_SD


if __name__ == "__main__":


    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    #sph kernel approximations
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    aperture = 30
    aperture_array = np.array([1, 3, 5, 10, 20, 30, 40, 50, 70, 100])
    _ap = np.where(aperture_array==aperture)[0]

    #For galaxies in region `num` and snap = tag
    num = str(ii)
    if len(num) == 1:
        num = '0'+num



    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder,num)
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    fl = flares.flares(fname = filename,sim_type = sim_type)


    S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len, BH_coords, S_ap, G_ap = get_data(num, tag, inp=inp, data_folder=data_folder)
    S_ap, G_ap = S_ap[_ap][0], G_ap[_ap][0]

    if len(S_len)==0:
        print (F"No data to write in region {num} for tag {tag}")
        S_los = np.zeros_like(S_len)
        BH_los = np.zeros_like(S_len)

    else:
        z = float(tag[5:].replace('p','.'))
        S_coords=S_coords.T/(1+z)
        G_coords=G_coords.T/(1+z)
        BH_coords=BH_coords.T/(1+z)

        sbegin, send = get_len(S_len)
        gbegin, gend = get_len(G_len)

        # print("calc shapes", sbegin.shape, send.shape, gbegin.shape, gend.shape)

        start = timeit.default_timer()
        pool = schwimmbad.MultiPool(processes=4)


        calc_Zlos = partial(get_ZLOS, req_coords=S_coords, G_coords=G_coords, G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, gbegin=gbegin, gend=gend, lkernel=lkernel, kbins=kbins, G_ap=G_ap, S_ap=S_ap, sbegin=sbegin, send=send)
        S_los = np.concatenate(np.array(list(pool.map(calc_Zlos, \
                                 np.arange(0,len(sbegin), dtype=np.int64)))))

        calc_Zlos = partial(get_ZLOS, req_coords=BH_coords, G_coords=G_coords, G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, gbegin=gbegin, gend=gend, lkernel=lkernel, kbins=kbins, G_ap=G_ap, Stars=False)
        BH_los = np.array(list(pool.map(calc_Zlos, \
                            np.arange(0, len(BH_coords), dtype=np.int64))))


        pool.close()
        stop = timeit.default_timer()
        print (F"Took {np.round(stop - start, 6)} seconds")


    print(F"Wrting out line-of-sight metal density of {tag} to {filename}")

    out_slos = np.zeros(len(S_coords))
    out_slos[S_ap] = S_los
    fl.create_dataset(out_slos, 'S_los', '{}/Particle'.format(tag),
        desc = F'Star particle line-of-sight metal column density along the z-axis within {aperture} pkpc',
        unit = 'Msun/pc^2', overwrite=True)

    fl.create_dataset(BH_los, 'BH_los', '{}/Particle'.format(tag),
        desc = F'BH particle line-of-sight metal column density along the z-axis within {aperture} pkpc', unit = 'Msun/pc^2', overwrite=True)
