import timeit, sys
import numpy as np
from mpi4py import MPI
from numba import jit, njit, float64, int32, prange
import h5py
from functools import partial
import schwimmbad
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u

import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)


@njit(float64[:](float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int32), parallel=True, nogil=True)
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

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


def get_data(ii, tag, inp = 'FLARES'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './data/FLARES_{}_sp_info.hdf5'.format(num)
        sim_type = 'FLARES'


    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./data/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    with h5py.File(filename, 'r') as hf:
        S_len = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_coords = np.array(hf[tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        G_mass = np.array(hf[tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml = np.array(hf[tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z = np.array(hf[tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)


    return S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len


def get_len(Length):

    begin = np.zeros(len(Length), dtype = np.int64)
    end = np.zeros(len(Length), dtype = np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)

    return begin, end


def get_ZLOS(jj, S_coords, G_coords, G_mass, G_Z, G_sml, sbegin, send, gbegin, gend, lkernel, kbins):

    this_scoords = S_coords[sbegin[jj]:send[jj]]
    this_gcoords = G_coords[gbegin[jj]:gend[jj]]

    this_gmass = G_mass[gbegin[jj]:gend[jj]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]]
    this_gsml = G_sml[gbegin[jj]:gend[jj]]


    Z_los_SD = get_Z_LOS(this_scoords, this_gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins)*conv

    return Z_los_SD


if __name__ == "__main__":


    ii, tag, inp = sys.argv[1], sys.argv[2], sys.argv[3]

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #sph kernel approximations
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']


    #For galaxies in region `num` and snap = tag
    num = str(ii)
    if len(num) == 1:
        num = '0'+num
    S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len = get_data(num, tag, inp = inp)
    z = float(tag[5:].replace('p','.'))
    S_coords=S_coords.T/(1+z)
    G_coords=G_coords.T/(1+z)

    total = np.arange(0, len(S_len))

    comm.Barrier()

    part = int(len(S_len)/size)
    if rank!=size-1:
        thisok = total[rank*part:(rank+1)*part]
    else:
        thisok = total[rank*part:]

    sbegin, send = get_len(S_len)
    gbegin, gend = get_len(G_len)

    S_coords = S_coords[sbegin[thisok[0]]:send[thisok[-1]]]

    G_coords = G_coords[gbegin[thisok[0]]:gend[thisok[-1]]]
    G_mass = G_mass[gbegin[thisok[0]]:gend[thisok[-1]]]
    G_Z = G_Z[gbegin[thisok[0]]:gend[thisok[-1]]]
    G_sml = G_sml[gbegin[thisok[0]]:gend[thisok[-1]]]


    if rank!=size-1:
        S_len = S_len[thisok[0]:thisok[-1]]
        G_len = G_len[thisok[0]:thisok[-1]]
    else:
        S_len = S_len[thisok[0]:]
        G_len = G_len[thisok[0]:]

    sbegin, send = get_len(S_len)
    gbegin, gend = get_len(G_len)

    start = timeit.default_timer()
    calc_Zlos = partial(get_ZLOS, S_coords=S_coords, G_coords=G_coords, G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, sbegin=sbegin, send=send, gbegin=gbegin, gend=gend, lkernel=lkernel, kbins=kbins)
    pool = schwimmbad.MultiPool(processes=4)
    tZlos = np.concatenate(np.array(list(pool.map(calc_Zlos, np.arange(0,len(sbegin))))))
    pool.close()
    stop = timeit.default_timer()
    print (F"Took {np.round(stop - start, 6)} seconds for rank = {rank}")

    comm.Barrier()

    if rank == 0:

        print ("Gathering data from different processes")

    Zlos = comm.gather(tZlos, root=0)

    if rank == 0:

        print ("Gathering completed")

        Zlos = np.concatenate(np.array(Zlos))
        if inp == 'FLARES':
            if len(num) == 1:
                num =  '0'+num
            filename = 'data2/FLARES_{}_sp_info.hdf5'.format(num)
            sim_type = 'FLARES'


        elif inp == 'REF' or inp == 'AGNdT9':
            filename = F"EAGLE_{inp}_sp_info.hdf5"
            sim_type = 'PERIODIC'

        print(F"Wrting out line-of-sight metal density to {filename}")
        print (Zlos)

        fl = flares.flares(fname = filename,sim_type = sim_type)

        fl.create_dataset(Zlos, 'S_los', '{}/Particle'.format(tag),
            desc = 'Star particle line-of-sight metal column density along the z-axis', unit = 'Msun/pc^2', overwrite=True)
