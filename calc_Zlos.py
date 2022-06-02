"""
    Calculates the line-of-sight metal density for star and SMBH particles using gas particles within a given aperture. At the moment the default is 30 pkpc same as the original EAGLE prescription
"""

import timeit, sys
import numpy as np
from numba import jit, njit, float64, int32, prange
import h5py
from functools import partial
import schwimmbad
from scipy.spatial import cKDTree
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u

import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)
norm = np.linalg.norm

@njit(float64[:](float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int32), parallel=True, nogil=True)
def old_cal_ZLOS(cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

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
    n = len(cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis.
    xdir, ydir, zdir = 0, 1, 2
    for ii in prange(n):

        thispos = cood[ii]
        ok = np.where(g_cood[:,zdir] > thispos[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thispos[xdir]
        y = thisgpos[:,ydir] - thispos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2


    return Z_los_SD

def cal_ZLOS_kd(req_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                 dimens=(0, 1, 2)):
    """

    Compute the los metal surface density (in Msun/Mpc^2) for given
    particles inside the galaxy taking the z-axis as the los. Method used
    in Roper+2022

    Args:
        req_cood (3d array): particle coordinates to calculate
                             metal line of sight density for
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length
        dimens (tuple: int): tuple of xyz coordinates

    """

    # Generalise dimensions (function assume LOS along z-axis)
    xdir, ydir, zdir = dimens

    # Get how many particles
    n = req_cood.shape[0]

    # Lets build the kd tree from star positions
    tree = cKDTree(req_cood[:, (xdir, ydir)])

    # Query the tree for all gas particles (can now supply multiple rs!)
    query = tree.query_ball_point(g_cood[:, (xdir, ydir)], r=g_sml, p=1)

    # Now we just need to collect each particle neighbours
    gas_nbours = {p: [] for p in range(n)}
    for g_ind, parts in enumerate(query):
        for _ind in parts:
            gas_nbours[_ind].append(g_ind)

    # Initialise line of sight metal density
    Z_los_SD = np.zeros(n)

    # Loop over the required particles
    for s_ind in range(n):

        # Extract gas particles to consider
        g_inds = gas_nbours.pop(s_ind)

        # Extract data for these particles
        thisspos = req_cood[s_ind]
        thisgpos = g_cood[g_inds]
        thisgsml = g_sml[g_inds]
        thisgZ = g_Z[g_inds]
        thisgmass = g_mass[g_inds]

        # We only want to consider particles "in-front" of the star
        ok = np.where(thisgpos[:, zdir] > thisspos[zdir])[0]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgZ = thisgZ[ok]
        thisgmass = thisgmass[ok]

        # Get radii and divide by smooting length
        b = np.linalg.norm(thisgpos[:, (xdir, ydir)]
                           - thisspos[((xdir, ydir), )],
                           axis=-1)
        boverh = b / thisgsml

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh])

        # Finally get LOS metal surface density in units of Msun/pc^2
        Z_los_SD[s_ind] = np.sum((thisgmass * thisgZ
                                  / (thisgsml * thisgsml))
                                 * kernel_vals)

    return Z_los_SD


def get_data(ii, tag, inp = 'FLARES', data_folder='data/', aperture=30):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        sim_type = 'FLARES'


    elif (inp == 'REF') or (inp == 'AGNdT9') or ('RECAL' in inp):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    print (tag, filename)

    with h5py.File(filename, 'r') as hf:
        S_len   = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len   = np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        BH_len  = np.array(hf[tag+'/Galaxy'].get('BH_Length'), dtype = np.int64)

        S_coords    = np.array(hf[tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords    = np.array(hf[tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        G_mass      = np.array(hf[tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml       = np.array(hf[tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z         = np.array(hf[tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)
        BH_coords   = np.array(hf[tag+'/Particle'].get('BH_Coordinates'), dtype = np.float64)

        S_ap = np.array(hf[tag+'/Particle/Apertures/Star'].get(F'{aperture}'), dtype = np.bool)
        G_ap = np.array(hf[tag+'/Particle/Apertures/Gas'].get(F'{aperture}'), dtype = np.bool)
        BH_ap = np.array(hf[tag+'/Particle/Apertures/BH'].get(F'{aperture}'), dtype = np.bool)


    return S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len, BH_len, BH_coords, S_ap, G_ap, BH_ap


def get_len(Length):

    begin = np.zeros(len(Length), dtype = np.int64)
    end = np.zeros(len(Length), dtype = np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)

    return begin, end


def get_ZLOS(jj, req_coords, begin, end, ap, G_coords, G_mass, G_Z, G_sml, gbegin, gend, lkernel, kbins, G_ap):

    this_coords = req_coords[begin[jj]:end[jj]][ap[begin[jj]:end[jj]]]

    this_gcoords = G_coords[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gmass = G_mass[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gsml = G_sml[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]


    Z_los_SD = cal_ZLOS_kd(this_coords, this_gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins)*conv

    return Z_los_SD


if __name__ == "__main__":


    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    #sph kernel approximations
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    aperture = 30

    #For galaxies in region `num` and snap = tag
    num = str(ii)
    if len(num) == 1:
        num = '0'+num



    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder,num)
        sim_type = 'FLARES'

    elif (inp == 'REF') or (inp == 'AGNdT9') or ('RECAL' in inp):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    else:
        ValueError("Type of input simulation not recognized")

    fl = flares.flares(fname = filename,sim_type = sim_type)

    S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len, BH_len, BH_coords, S_ap, G_ap, BH_ap = get_data(num, tag, inp=inp, data_folder=data_folder, aperture=aperture)

    if len(S_len)==0:
        print (F"No data to write in region {num} for tag {tag}")
        S_los = np.zeros_like(S_len)
        BH_los = np.zeros_like(BH_len)

    else:
        z = float(tag[5:].replace('p','.'))
        S_coords=S_coords.T/(1+z)
        G_coords=G_coords.T/(1+z)
        BH_coords=BH_coords.T/(1+z)

        sbegin, send = get_len(S_len)
        bhbegin, bhend = get_len(BH_len)
        gbegin, gend = get_len(G_len)

        # print("calc shapes", sbegin.shape, send.shape, gbegin.shape, gend.shape)

        start = timeit.default_timer()
        pool = schwimmbad.SerialPool()


        calc_Zlos = partial(get_ZLOS, req_coords=S_coords, ap=S_ap, begin=sbegin, end=send, G_coords=G_coords, G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, gbegin=gbegin, gend=gend, lkernel=lkernel, kbins=kbins, G_ap=G_ap)
        S_los = np.concatenate(np.array(list(pool.map(calc_Zlos, \
                                 np.arange(0,len(sbegin), dtype=np.int64)))))

        calc_Zlos = partial(get_ZLOS, req_coords=BH_coords, ap=BH_ap, begin=bhbegin, end=bhend, G_coords=G_coords, G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, gbegin=gbegin, gend=gend, lkernel=lkernel, kbins=kbins, G_ap=G_ap)
        BH_los = np.concatenate(np.array(list(pool.map(calc_Zlos, \
                            np.arange(0, len(bhbegin), dtype=np.int64)))))


        pool.close()
        stop = timeit.default_timer()
        print (F"Took {np.round(stop - start, 6)} seconds")


    print(F"Wrting out line-of-sight metal density of {tag} to {filename}")

    out_slos = np.zeros(len(S_coords))
    out_slos[S_ap] = S_los
    fl.create_dataset(out_slos, 'S_los', '{}/Particle'.format(tag),
        desc = F'Star particle line-of-sight metal column density along the z-axis within {aperture} pkpc',
        unit = 'Msun/pc^2', overwrite=True)

    out_bhlos = np.zeros(len(BH_coords))
    out_bhlos[BH_ap] = BH_los
    fl.create_dataset(out_bhlos, 'BH_los', '{}/Particle'.format(tag),
        desc = F'BH particle line-of-sight metal column density along the z-axis within {aperture} pkpc', unit = 'Msun/pc^2', overwrite=True)
