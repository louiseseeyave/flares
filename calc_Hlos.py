"""
    Calculates the line-of-sight metal density for star and SMBH particles using gas particles within a given aperture. At the moment the default is 30 pkpc same as the original EAGLE prescription
"""

import timeit, sys
import numpy as np
# from numba import jit, njit, float64, int32, prange
import h5py
from functools import partial
import schwimmbad
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from scipy.spatial import cKDTree

import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)
norm = np.linalg.norm

#@njit(float64[:], parallel=True, nogil=True)
def old_cal_HLOS(cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                 frac=False):

    """

    Compute the los metal surface density (in Msun/Mpc^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length
        lkernel: kernel look-up table
        kbins: number of bins in the look-up table

    """
    
    n = len(cood)
    print('no. of stellar particles:', n)
    H_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis.
    xdir, ydir, zdir = 0, 1, 2 # associate direction with column in coord array
    for ii in range(n):

        thispos = cood[ii]
        ok = np.where(g_cood[:,zdir] > thispos[zdir])[0] # remove gas particles behind star (z dir)
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        #thisgH = 1-g_Z # get hydrogen mass fraction (need He frac)
        thisgH = np.full(len(thisgpos), 0.75) # use this for now
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thispos[xdir]
        y = thisgpos[:,ydir] - thispos[ydir]

        b = np.sqrt(x*x + y*y) # get impact parameter
        boverh = b/thisgsml 

        ok = np.where(boverh <= 1.)[0] # want smoothing length > impact parameter
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        H_los_SD[ii] = np.sum((thisgmass[ok]*thisgH[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2
        #if frac==True:
        #    H_los_SD[ii] = np.sum((thisgH[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of /pc^2

    return H_los_SD


def new_cal_HLOS(cood, g_cood, g_mass, g_hfrac, g_temp, g_sml, lkernel,
                 kbins):

    """

    Compute the los hydrogen surface density (in Msun/Mpc^2) for star
    particles inside the galaxy, taking the z-axis as the los.
    Args:
        cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_hfrac (1d array): gas particle hydrogen mass fraction
        g_temp (1d array): gas particle temperature
        g_sml (1d array): gas particle smoothing length
        lkernel: kernel look-up table
        kbins: number of bins in the look-up table

    """

    # print('s_coords:', cood[0:5])
    # print('g_coords:', g_cood[0:5])
    # print('g_mass:', g_mass[0:5])
    # print('g_hfrac:', g_hfrac[0:5])
    # print('g_temp:', g_temp[0:5])
    # print('g_sml:', g_sml[0:5])

    # get array to store H column density
    n = len(cood)
    #print('no. of stellar particles:', n)
    H_los_SD = np.zeros(n)

    # associate direction with column in coord array
    xdir, ydir, zdir = 0, 1, 2

    for ii in range(n):

        # get H column density this stellar particle
        thispos = cood[ii]

        #print('no. of gas particles:', len(g_cood))

        # remove gas particles behind star (z dir)
        stellar_z = thispos[zdir]
        ok = g_cood[:,zdir] > stellar_z
        #print('len ok:', len(ok))

        #print('no. of gas particles after z cut:', np.sum(ok))
        
        # remove gas particles with temp > 1.5*10^4K (i.e. ionised)
        max_temp = 1.5*10**4
        ok = np.logical_and(ok, g_temp < max_temp)

        #print('no. of gas particles after temp cut:', np.sum(ok))
        #print('ok temps:', (g_temp[ok])[0:5])
        
        # get quantities for relevant gas particles
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgH = g_hfrac[ok]
        # thisgH = np.full(len(thisgpos), 0.75) #  old definition
        thisgmass = g_mass[ok]

        # get impact parameter (i.e. distance to stellar particle)
        x = thisgpos[:,xdir] - thispos[xdir]
        y = thisgpos[:,ydir] - thispos[ydir]
        b = np.sqrt(x*x + y*y)

        # impact parameter / smoothing length - to use in look-up table
        boverh = b/thisgsml 

        # remove gas particles if smoothing length < impact parameter
        ok = boverh <= 1.
        g_len = np.sum(ok)
        #print('g_len:', g_len)

        # get kernel values from look-up table
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])
        #print('kernel_vals:', kernel_vals[0:5])

        # calculate H column density [Msun/Mpc^2]
        H_los_SD[ii] = np.sum((thisgmass[ok]*thisgH[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals)

        #print('H_LOS range:', np.amin(H_los_SD[ii]), np.amax(H_los_SD[ii]))

    return H_los_SD
    #return H_los_SD, g_len, stellar_z


def cal_HLOS_kd(req_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                dimens=(0, 1, 2)):
    """

    Compute the los metal surface density (in Msun/Mpc^2) for given
    particles inside the galaxy taking the z-axis as the los. Method used
    in Roper+2022

    This method doesn't work for me. Get b/h too large for look-up table.

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
    H_los_SD = np.zeros(n)

    # Loop over the required particles
    for s_ind in range(n):

        # Extract gas particles to consider
        g_inds = gas_nbours.pop(s_ind)

        # Extract data for these particles
        thisspos = req_cood[_ind]
        thisgpos = g_cood[g_inds]
        thisgsml = g_sml[g_inds]
        #thisgZ = g_Z[g_inds]
        thisgH = np.full(len(thisgpos), 0.75) # use this for now
        thisgmass = g_mass[g_inds]

        # We only want to consider particles "in-front" of the star
        ok = np.where(thisgpos[:, zdir] > thisspos[zdir])[0]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgH = thisgH[ok]
        thisgmass = thisgmass[ok]

        # Get radii and divide by smoothing length
        b = np.linalg.norm(thisgpos[:, (xdir, ydir)]
                           - thisspos[((xdir, ydir), )],
                           axis=-1)
        boverh = b / thisgsml
        print('max boverh:', np.amax(boverh))
        print('min boverh:', np.amin(boverh))

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh])

        # Finally get LOS metal surface density in units of Msun/pc^2
        H_los_SD[s_ind] = np.sum((thisgmass * thisgH
                                  / (thisgsml * thisgsml))
                                 * kernel_vals)

    return H_los_SD


def get_data(ii, tag, inp = 'FLARES', data_folder='data/', aperture=30):

    # data_folder doesn't matter anymore
    
    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        #filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        #filename = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
        filename = '/cosma7/data/dp004/dc-seey1/data/flares/myversion/flares.hdf5'
        sim_type = 'FLARES'


    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
    print (filename)

    with h5py.File(filename, 'r') as hf:
        S_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        BH_len  = np.array(hf[num+'/'+tag+'/Galaxy'].get('BH_Length'), dtype = np.int64)

        S_coords    = np.array(hf[num+'/'+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords    = np.array(hf[num+'/'+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        G_mass      = np.array(hf[num+'/'+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml       = np.array(hf[num+'/'+tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z         = np.array(hf[num+'/'+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)
        G_H         = np.array(hf[num+'/'+tag+'/Particle'].get('H_frac_smooth'), dtype = np.float64)
        G_Temp      = np.array(hf[num+'/'+tag+'/Particle'].get('G_Temp'), dtype = np.float64)
        BH_coords   = np.array(hf[num+'/'+tag+'/Particle'].get('BH_Coordinates'), dtype = np.float64)

        S_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/Star'].get(F'{aperture}'), dtype = np.bool)
        G_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/Gas'].get(F'{aperture}'), dtype = np.bool)
        BH_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/BH'].get(F'{aperture}'), dtype = np.bool)


    return S_coords, G_coords, G_mass, G_sml, G_Z, G_H, G_Temp, S_len, G_len, BH_len, BH_coords, S_ap, G_ap, BH_ap


def get_len(Length):

    begin = np.zeros(len(Length), dtype = np.int64)
    end = np.zeros(len(Length), dtype = np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)

    return begin, end


def get_relevant_particles(ii, tag, inp = 'FLARES', data_folder='data/', aperture=30):

    """get data for particles involved in the calculation"""

    # data_folder doesn't matter anymore
    
    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        #filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        filename = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
        sim_type = 'FLARES'


    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
    print (filename)

    with h5py.File(filename, 'r') as hf:
        S_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len   = np.array(hf[num+'/'+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        BH_len  = np.array(hf[num+'/'+tag+'/Galaxy'].get('BH_Length'), dtype = np.int64)

        S_coords    = np.array(hf[num+'/'+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords    = np.array(hf[num+'/'+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        G_mass      = np.array(hf[num+'/'+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml       = np.array(hf[num+'/'+tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z         = np.array(hf[num+'/'+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)
        BH_coords   = np.array(hf[num+'/'+tag+'/Particle'].get('BH_Coordinates'), dtype = np.float64)

        S_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/Star'].get(F'{aperture}'), dtype = np.bool)
        G_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/Gas'].get(F'{aperture}'), dtype = np.bool)
        BH_ap = np.array(hf[num+'/'+tag+'/Particle/Apertures/BH'].get(F'{aperture}'), dtype = np.bool)

    def get_len(Length):

        begin = np.zeros(len(Length), dtype = np.int64)
        end = np.zeros(len(Length), dtype = np.int64)
        begin[1:] = np.cumsum(Length)[:-1]
        end = np.cumsum(Length)

        return begin, end

    sbegin, send = get_len(S_len)
    #bhbegin, bhend = get_len(BH_len)
    gbegin, gend = get_len(G_len)


    this_coords = req_coords[sbegin[jj]:send[jj]][ap[sbegin[jj]:send[jj]]]

    this_gcoords = G_coords[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gmass = G_mass[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gsml = G_sml[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    
    return this_coords, this_gcoords, this_gmass, this_gZ, this_gsml
    

def get_HLOS(jj, req_coords, begin, end, ap, G_coords, G_mass, G_Z, G_H,
             G_Temp, G_sml, gbegin, gend, lkernel, kbins, G_ap):

    #print('len req_coords', len(req_coords))
    #print(len(req_coords[begin[jj]:end[jj]]))
    #print(len(req_coords[begin[jj]:end[jj]][ap[begin[jj]:end[jj]]]))

    # stellar particle coordinates
    this_coords = req_coords[begin[jj]:end[jj]][ap[begin[jj]:end[jj]]]

    # gas particle quantities
    this_gcoords = G_coords[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gmass = G_mass[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gH = G_H[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gtemp = G_Temp[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]
    this_gsml = G_sml[gbegin[jj]:gend[jj]][G_ap[gbegin[jj]:gend[jj]]]

    # H_los_SD = cal_HLOS_kd(this_coords, this_gcoords, this_gmass, this_gZ,
    #                        this_gsml, lkernel, kbins)*conv
    # H_los_SD = old_cal_HLOS(this_coords, this_gcoords, this_gmass, this_gZ,
    #                         this_gsml, lkernel, kbins, frac=frac)*conv

    H_los_SD = new_cal_HLOS(this_coords, this_gcoords, this_gmass, this_gH,
                            this_gtemp, this_gsml, lkernel, kbins)*conv

    #print(len(H_los_SD))
    return H_los_SD


if __name__ == "__main__":


    # UNCOMMENT THIS IF USING SBATCH
    # ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    #
    # test:
    ii, tag, inp, data_folder = 0, '010_z005p000', 'FLARES', 'data'
    
    # inp & data_folder don't matter anymore

    #sph kernel approximations
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    print('lkernel:', lkernel)
    header = kinp['header']
    kbins = header.item()['bins']

    aperture = 30

    #For galaxies in region `num` and snap = tag
    num = str(ii)
    if len(num) == 1:
        num = '0'+num

    #if inp == 'FLARES':
    #    if len(num) == 1:
    #        num =  '0'+num
    #    filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder,num)
    #    sim_type = 'FLARES'

    #elif inp == 'REF' or inp == 'AGNdT9':
    #    filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
    #    sim_type = 'PERIODIC'

    #fl = flares.flares(fname = filename,sim_type = sim_type)

    # regions = np.array(['00','01','02','03','04','05','06','07','08','09',
    #                     '10','11','12','13','14','15','16','17','18','19',
    #                     '20','21','22','23','24','25','26','27','28','29',
    #                     '30','31','32','33','34','35','36','37','38','39'])


    # S_coords = []
    # G_coords = []
    # G_mass = []
    # G_sml = []
    # G_Z = []
    # S_len = []
    # G_len = []
    # BH_len = []
    # BH_coords = []
    # S_ap = []
    # G_ap = []
    # BH_ap = []
    # for rg in regions:
        # xS_coords, xG_coords, xG_mass, xG_sml, xG_Z, xS_len, xG_len, xBH_len, xBH_coords, xS_ap, xG_ap, xBH_ap = get_data(rg, tag, inp=inp, data_folder=data_folder, aperture=aperture)
        # S_coords.extend(xS_coords)
        # G_coords.extend(xG_coords)
        # G_mass.extend(xG_mass)
        # G_sml.extend(xG_sml)
        # G_Z.extend(G_Z)
        # S_len.extend(xS_len)
        # G_len.extend(xG_len)
        # BH_len.extend(xBH_len)
        # BH_coords.extend(xBH_coords)
        # S_ap.extend(xS_ap)
        # G_ap.extend(xG_ap)
        # BH_ap.extend(xBH_ap)

    S_coords, G_coords, G_mass, G_sml, G_Z, G_H, G_Temp, S_len, G_len, BH_len, BH_coords, S_ap, G_ap, BH_ap = get_data(ii, tag, inp=inp, data_folder=data_folder, aperture=aperture)


    # debugging stuff ------------------------------------------------
    print('S_coords:')
    #print(np.shape(S_coords))
    #S_coords = np.transpose(S_coords)
    print(np.shape(S_coords))
    print(np.shape(S_coords[2]))
    print(np.amin(S_coords[2]), np.amax(S_coords[2]))

    print('G_coords:')
    #print(np.shape(S_coords))
    #S_coords = np.transpose(S_coords)
    print(np.shape(G_coords))
    print(np.shape(G_coords[2]))
    print(np.amin(G_coords[2]), np.amax(G_coords[2]))

    print('G_sml:')
    #print(np.shape(S_coords))
    #S_coords = np.transpose(S_coords)
    print(np.shape(G_sml))
    print(np.amin(G_sml), np.amax(G_sml))

    print('G_mass:')
    #print(np.shape(S_coords))
    #S_coords = np.transpose(S_coords)
    print(np.shape(G_mass))
    print(np.amin(G_mass), np.amax(G_mass))
    
    print('G_H:')
    print(np.shape(G_H))
    print(G_H[0:5])

    print('G_Temp:')
    print(np.shape(G_Temp))
    print(G_Temp[0:5])
    
    # debugging stuff ------------------------------------------------
    
    # S_coords, G_coords, G_mass, G_sml, G_Z, S_len, G_len, BH_len, BH_coords, S_ap, G_ap, BH_ap = get_data(ii, tag, inp=inp, data_folder=data_folder, aperture=aperture)

    print('sum S_len:', np.sum(S_len))
    print('len S_ap:', len(S_ap))
    print('len S_coords:', len(S_coords))

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

        print('len S_coords:', len(S_coords))
        # print("calc shapes", sbegin.shape, send.shape, gbegin.shape, gend.shape)

        start = timeit.default_timer()
        pool = schwimmbad.SerialPool()

        # debugging code
        #calc_Hlos = get_HLOS(0, req_coords=S_coords, ap=S_ap, begin=sbegin,
        #                    end=send, G_coords=G_coords, G_mass=G_mass,
        #                    G_Z=G_Z, G_H=G_H, G_Temp=G_Temp, G_sml=G_sml,
        #                    gbegin=gbegin, gend=gend, lkernel=lkernel,
         #                   kbins=kbins, G_ap=G_ap)
        
        # --------------

        calc_Hlos = partial(get_HLOS, req_coords=S_coords, ap=S_ap, begin=sbegin,
                            end=send, G_coords=G_coords, G_mass=G_mass,
                            G_Z=G_Z, G_H=G_H, G_Temp=G_Temp, G_sml=G_sml,
                            gbegin=gbegin, gend=gend, lkernel=lkernel,
                            kbins=kbins, G_ap=G_ap)
        S_los = np.concatenate(np.array(list(pool.map(calc_Hlos, \
                                 np.arange(0,len(sbegin), dtype=np.int64)))))

        # calc_Hlos = partial(get_HLOS, req_coords=BH_coords, ap=BH_ap,
        #                     begin=bhbegin, end=bhend, G_coords=G_coords,
        #                     G_mass=G_mass, G_Z=G_Z, G_sml=G_sml, gbegin=gbegin,
        #                     gend=gend, lkernel=lkernel, kbins=kbins,
        #                     G_ap=G_ap, frac=H_frac)
        # BH_los = np.concatenate(np.array(list(pool.map(calc_Hlos, \
        #                     np.arange(0, len(bhbegin), dtype=np.int64)))))


        pool.close()
        stop = timeit.default_timer()
        print (F"Took {np.round(stop - start, 6)} seconds")

    exit() # -----------------------------------
    print(F"Writing out line-of-sight metal density of {tag}")

    # THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    out_slos = np.zeros(len(S_coords))
    out_slos[S_ap] = S_los

    # out_bhlos = np.zeros(len(BH_coords))
    # out_bhlos[BH_ap] = BH_los

    label = 'new'
    
    with h5py.File(f'/cosma7/data/dp004/dc-seey1/data/flares/temp/flares_{num}.hdf5', 'a') as fl:

        fl.create_dataset(f'{num}/{tag}/Particle/S_Hlos_{label}', data = out_slos)
        fl[f'{num}/{tag}/Particle/S_Hlos_{label}'].attrs['Description'] = f'Star particle line-of-sight H column density along the z-axis within {aperture} pkpc'
        fl[f'{num}/{tag}/Particle/S_Hlos_{label}'].attrs['Units'] = 'Msun/pc^2'
        
        #fl.create_dataset(f'{num}/{tag}/Particle/BH_{H_label}los', data = out_bhlos)
        #fl[f'{num}/{tag}/Particle/BH_{H_label}los'].attrs['Description'] = f'BH particle line-of-sight {H_label} column density along the z-axis within {aperture} pkpc'
        #fl[f'{num}/{tag}/Particle/BH_{H_label}los'].attrs['Units'] = H_units
        
