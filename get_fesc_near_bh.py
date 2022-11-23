import sys
import numpy as np
import eagle_IO.eagle_IO as E
import pickle
from calc_Hlos import new_cal_HLOS
import h5py
from scipy.spatial import cKDTree

from astropy import units as u
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)


# for investigating the black hole environment


def save_data(newfile, rg, snap, nsf_bhmass, nsf_bhids, nstellar_5pkpc,
              mstellar_5pkpc, ngas_5pkpc, mgas_5pkpc, mean_fesc_5pkpc):

    print(f'saving data in {newfile}...')

    with h5py.File(newfile, 'a') as fl:

        # delete datasets if they exist ------------------------------
        
        if f'{rg}/{snap}/Particle/BH_Mass_notsubfind' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_Mass_notsubfind')
            del fl[f'{rg}/{snap}/Particle/BH_Mass_notsubfind']

        if f'{rg}/{snap}/Particle/BH_ID_notsubfind' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_ID_notsubfind')
            del fl[f'{rg}/{snap}/Particle/BH_ID_notsubfind']

        if f'{rg}/{snap}/Particle/BH_nstellar_5pkpc' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_nstellar_5pkpc')
            del fl[f'{rg}/{snap}/Particle/BH_nstellar_5pkpc']

        if f'{rg}/{snap}/Particle/BH_mstellar_5pkpc' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_mstellar_5pkpc')
            del fl[f'{rg}/{snap}/Particle/BH_mstellar_5pkpc']

        if f'{rg}/{snap}/Particle/BH_ngas_5pkpc' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_ngas_5pkpc')
            del fl[f'{rg}/{snap}/Particle/BH_ngas_5pkpc']

        if f'{rg}/{snap}/Particle/BH_mgas_5pkpc' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_mgas_5pkpc')
            del fl[f'{rg}/{snap}/Particle/BH_mgas_5pkpc']

        if f'{rg}/{snap}/Particle/BH_meanfesc_5pkpc' in fl:
            print(f'deleting {rg}/{snap}/Particle/BH_meanfesc_5pkpc')
            del fl[f'{rg}/{snap}/Particle/BH_meanfesc_5pkpc']

        # create datasets --------------------------------------------
        
        fl.create_dataset(f'{rg}/{snap}/Particle/BH_Mass_notsubfind', data = nsf_bhmass)
        fl[f'{rg}/{snap}/Particle/BH_Mass_notsubfind'].attrs['Description'] = 'Black hole mass - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_Mass_notsubfind'].attrs['Units'] = '1E10 Msun'

        fl.create_dataset(f'{rg}/{snap}/Particle/BH_ID_notsubfind', data = nsf_bhids)
        fl[f'{rg}/{snap}/Particle/BH_ID_notsubfind'].attrs['Description'] = 'Black hole particle ID - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_ID_notsubfind'].attrs['Units'] = 'None'
        
        fl.create_dataset(f'{rg}/{snap}/Particle/BH_nstellar_5pkpc', data = nstellar_5pkpc)
        fl[f'{rg}/{snap}/Particle/BH_nstellar_5pkpc'].attrs['Description'] = 'No. of stellar particles within 5pkpc of BH - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_nstellar_5pkpc'].attrs['Units'] = 'None'

        fl.create_dataset(f'{rg}/{snap}/Particle/BH_mstellar_5pkpc', data = mstellar_5pkpc)
        fl[f'{rg}/{snap}/Particle/BH_mstellar_5pkpc'].attrs['Description'] = 'Mass of stellar particles within 5pkpc of BH - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_mstellar_5pkpc'].attrs['Units'] = 'None'

        fl.create_dataset(f'{rg}/{snap}/Particle/BH_ngas_5pkpc', data = ngas_5pkpc)
        fl[f'{rg}/{snap}/Particle/BH_ngas_5pkpc'].attrs['Description'] = 'No. of gas particles within 5pkpc of BH - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_ngas_5pkpc'].attrs['Units'] = 'None'

        fl.create_dataset(f'{rg}/{snap}/Particle/BH_mgas_5pkpc', data = mgas_5pkpc)
        fl[f'{rg}/{snap}/Particle/BH_mgas_5pkpc'].attrs['Description'] = 'Mass of gas particles within 5pkpc of BH - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_mgas_5pkpc'].attrs['Units'] = 'None'

        fl.create_dataset(f'{rg}/{snap}/Particle/BH_meanfesc_5pkpc', data = mean_fesc_5pkpc)
        fl[f'{rg}/{snap}/Particle/BH_meanfesc_5pkpc'].attrs['Description'] = 'Mean escape fraction of stellar particls within 5pkpc of BH - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/BH_meanfesc_5pkpc'].attrs['Units'] = 'None'

        

if __name__ == "__main__":

    ii, snap = sys.argv[1], sys.argv[2]

    if len(str(ii)) == 1:
        rg = '0' + str(ii)
    else:
        rg = str(ii)

    print(f'region {rg}, snapshot {snap}')

    # load masks -----------------------------------------------------

    with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc.pkl', 'rb') as f:
        nbours = pickle.load(f)
        s_nbours = nbours['stellar_neighbours']
        g_nbours = nbours['gas_neighbours']
        bh_nbours = nbours['bh_neighbours']

    if s_nbours=={}:
        print('no galaxies, saving empty arrays...')
        newfile = f'/cosma7/data/dp004/dc-seey1/data/flares/temp/flares_{rg}.hdf5'
        nstellar_5pkpc = []
        mstellar_5pkpc = []
        ngas_5pkpc = []
        mgas_5pkpc = []
        mean_fesc_5pkpc = []
        nsf_bhids = []
        nsf_bhmass = []
        save_data(newfile, rg, snap, nsf_bhmass, nsf_bhids, nstellar_5pkpc,
              mstellar_5pkpc, ngas_5pkpc, mgas_5pkpc, mean_fesc_5pkpc)
        exit()

    # load s_hlos data from master ------------------------------------

    fname = '/cosma7/data/dp004/dc-seey1/data/flares/scripts/flares.hdf5'
    m = h5py.File(fname, 'r')
    s_length = np.array(m[f'{rg}/{snap}/Galaxy/S_Length_notsubfind'], dtype=np.int64)
    s_begin = np.zeros(len(s_length), dtype=np.int64)
    s_end = np.zeros(len(s_length), dtype=np.int64)
    s_begin[1:] = np.cumsum(s_length)[:-1]
    s_end = np.cumsum(s_length)
    s_hlos = np.array(m[f'{rg}/{snap}/Particle/S_Hlos_notsubfind'], dtype=np.float64)
    
    # load particle data ---------------------------------------------

    nThreads = 8
    sim = f'/cosma7/data/dp004/FLARES/FLARES-1/flares_{rg}/data'
    s_coords = E.read_array('PARTDATA', sim, snap, '/PartType4/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    g_coords = E.read_array('PARTDATA', sim, snap, '/PartType0/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    try:
        bh_coords = E.read_array('PARTDATA', sim, snap, '/PartType5/Coordinates',
                                 noH=True, physicalUnits=True, numThreads=nThreads,
                                 CGS=False)
    except ValueError:
        print('no black holes in partdata, saving empty arrays...')
        newfile = f'/cosma7/data/dp004/dc-seey1/data/flares/temp/flares_{rg}.hdf5'
        nstellar_5pkpc = []
        mstellar_5pkpc = []
        ngas_5pkpc = []
        mgas_5pkpc = []
        mean_fesc_5pkpc = []
        nsf_bhids = []
        nsf_bhmass = []
        save_data(newfile, rg, snap, nsf_bhmass, nsf_bhids, nstellar_5pkpc,
              mstellar_5pkpc, ngas_5pkpc, mgas_5pkpc, mean_fesc_5pkpc)
        exit()

    s_coords = np.array(s_coords, dtype=np.float64)
    g_coords = np.array(g_coords, dtype=np.float64)
    bh_coords = np.array(bh_coords, dtype=np.float64)
    
    print('s, g, bh coord shapes:', s_coords.shape, g_coords.shape, bh_coords.shape)

    bh_ids = E.read_array('PARTDATA', sim, snap, '/PartType5/ParticleIDs',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    bh_mass = E.read_array('PARTDATA', sim, snap, '/PartType5/BH_Mass',
                           noH=True, physicalUnits=True, numThreads=nThreads,
                           CGS=False)
    bh_ids = np.array(bh_ids, dtype=np.int64)
    bh_mass = np.array(bh_mass, dtype=np.float64)

    s_mass = E.read_array('PARTDATA', sim, snap, '/PartType4/Mass',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    g_mass = E.read_array('PARTDATA', sim, snap, '/PartType0/Mass',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    s_mass = np.array(s_mass, dtype=np.float64)
    g_mass = np.array(g_mass, dtype=np.float64)

    # keep only the particles inside each galaxy ---------------------

    nstellar_5pkpc = []
    mstellar_5pkpc = np.array([])
    ngas_5pkpc = []
    mgas_5pkpc = np.array([])
    mean_fesc_5pkpc = []
    nsf_bhids = np.array([])
    nsf_bhmass = np.array([])
    
    for ii in range(len(g_nbours)): # for each galaxy

        # apply masks
        s_keep = s_nbours[ii]
        print(f'there are {len(s_keep)} stellar particles in this galaxy')
        g_keep = g_nbours[ii]
        print(f'there are {len(g_keep)} gas particles in this galaxy')
        bh_keep = bh_nbours[ii]
        print(f'there are {len(bh_keep)} bh particles in this galaxy')

        gal_scoords = s_coords[s_keep]
        gal_smass = s_mass[s_keep]
        gal_gcoords = g_coords[g_keep]
        gal_gmass = g_mass[g_keep]
        gal_bhcoords = bh_coords[bh_keep]
        gal_bhids = bh_ids[bh_keep]
        gal_bhmass = bh_mass[bh_keep]

        gal_shlos = s_hlos[s_begin[ii]:s_end[ii]] # [Msun/Mpc^2]

        nsf_bhids = np.append(nsf_bhids, gal_bhids)
        nsf_bhmass = np.append(nsf_bhmass, gal_bhmass)

        # find particles within 5pkpc of each bh ---------------------

        # build kd tree from bh coords
        print('building kd tree from bh coords...')
        tree = cKDTree(gal_bhcoords)

        # query it for stellar and gas particles
        print('querying for stellar and gas particles...')
        s_query = tree.query_ball_point(gal_scoords, r=0.005, p=2)
        g_query = tree.query_ball_point(gal_gcoords, r=0.005, p=2)

        # collect particles in each galaxy ---------------------------
        n = len(gal_bhcoords)
        print(f'there are {n} black holes to group particles into')
        s_nbours_5pkpc = {g: [] for g in range(n)}
        g_nbours_5pkpc = {g: [] for g in range(n)}

        print('grouping stellar particles...')
        for s_ind, parts in enumerate(s_query):
            for _ind in parts:
                s_nbours_5pkpc[_ind].append(s_ind)

        print('grouping gas particles...')
        for g_ind, parts in enumerate(g_query):
            for _ind in parts:
                g_nbours_5pkpc[_ind].append(g_ind)

        # get quantities (5pkpc aperture around black hole) ----------

        for bh in s_nbours_5pkpc:

            sp = s_nbours_5pkpc[bh]
            nstellar_5pkpc.append(len(sp))
            smass = gal_smass[sp]
            mstellar_5pkpc = np.append(mstellar_5pkpc, smass)

            gp = g_nbours_5pkpc[bh]
            ngas_5pkpc.append(len(gp))
            gmass = gal_gmass[gp]
            mgas_5pkpc = np.append(mgas_5pkpc, gmass)

            bh_shlos = gal_shlos[sp]
            M_sun = 1.99 * 10**30 # [kg]
            M_H = 1.67 * 10**-27 # [kg]
            bh_shlos = bh_shlos * (M_sun/M_H) # [H atoms/pc^2]
            sigma = 6.3 * 10**-18 # photoionisation cross section [cm^2]
            sigma = sigma / (3.1*10**18)**2 # [pc^2]
            fesc = np.mean(np.exp(-sigma*s_hlos))
            mean_fesc_5pkpc.append(fesc)


    newfile = f'/cosma7/data/dp004/dc-seey1/data/flares/temp/flares_{rg}.hdf5'
    save_data(newfile, rg, snap, nsf_bhmass, nsf_bhids, nstellar_5pkpc,
              mstellar_5pkpc, ngas_5pkpc, mgas_5pkpc, mean_fesc_5pkpc)



    print('done for this round!')
