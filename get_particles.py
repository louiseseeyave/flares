import sys
import numpy as np
import eagle_IO.eagle_IO as E
import h5py
from scipy.spatial import cKDTree
import time

# we take the COP of each galaxy and identify all particles within
# a 30 pkpc radius of it

if __name__ == "__main__":

    print('got here')
    ii, snap = sys.argv[1], sys.argv[2]

    if len(str(ii)) == 1:
        rg = '0' + str(ii)
    else:
        rg = str(ii)

    print(f'region {rg}, snapshot {snap}')

    # get COP of all galaxies in that region
    master = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
    with h5py.File(master, 'r') as m:
        cop = np.array(m[f'{rg}/{snap}/Galaxy/COP'])
    print('cop shape:', cop.shape)
    cop = np.transpose(cop)
    print('new cop shape:', cop.shape)

    # get particle coordinates [cMpc]
    nThreads = 8
    sim = f'/cosma7/data/dp004/FLARES/FLARES-1/flares_{rg}/data'
    s_coords = E.read_array('PARTDATA', sim, snap, '/PartType4/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    g_coords = E.read_array('PARTDATA', sim, snap, '/PartType0/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    bh_coords = E.read_array('PARTDATA', sim, snap, '/PartType5/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    s_coords = np.array(s_coords, dtype=np.float64)
    g_coords = np.array(g_coords, dtype=np.float64)
    bh_coords = np.array(bh_coords, dtype=np.float64)

    print('s, g, bh coord shapes:', s_coords.shape, g_coords.shape, bh_coords.shape)

    # I DON'T THINK THIS IS NEEDED
    # get particle smoothing lengths [pMpc]
    # s_sml = E.read_array('PARTDATA', sim, snap, '/PartType4/SmoothingLength',
    #                         noH=True, physicalUnits=True, numThreads=nThreads,
    #                         CGS=False)
    # g_sml = E.read_array('PARTDATA', sim, snap, '/PartType0/SmoothingLength',
    #                         noH=True, physicalUnits=True, numThreads=nThreads,
    #                         CGS=False)
    # bh_sml = E.read_array('PARTDATA', sim, snap, '/PartType5/SmoothingLength',
    #                         noH=True, physicalUnits=True, numThreads=nThreads,
    #                         CGS=False)
    # s_sml = np.array(s_sml)
    # g_sml = np.array(g_sml)
    # bh_sml = np.array(bh_sml)

    # get particle IDs
    # s_ids = E.read_array('PARTDATA', sim, snap, '/PartType4/ParticleIDs',
    #                      noH=True, physicalUnits=True, numThreads=nThreads,
    #                      CGS=False)
    # g_ids = E.read_array('PARTDATA', sim, snap, '/PartType0/ParticleIDs',
    #                      noH=True, physicalUnits=True, numThreads=nThreads,
    #                      CGS=False)
    # bh_ids = E.read_array('PARTDATA', sim, snap, '/PartType5/ParticleIDs',
    #                       noH=True, physicalUnits=True, numThreads=nThreads,
    #                       CGS=False)
    # s_ids = np.array(s_ids, dtype=np.int64)
    # g_ids = np.array(g_coords, dtype=np.int64)
    # bh_ids = np.array(bh_coords, dtype=np.int64)

    # print('s, g, bh ids shapes:', s_ids.shape, g_ids.shape, bh_ids.shape)
    
    # convert comoving coordinates to physical
    if len(cop)==0:
        print (f"No galaxies in region {rg} for snapshot {snap}")
    else:
        z = float(snap[5:].replace('p','.'))
        print(f'redshift {z}, converting to physical coordinates...')
        s_coords=s_coords/(1+z)
        g_coords=g_coords/(1+z)
        bh_coords=bh_coords/(1+z)


    # WHICH METHOD SHOULD I USE... THIS:
        
    # build kd tree from COPs
    # s_tree = cKDTree(s_coords)
    # g_tree = cKDTree(g_coords)
    # bh_tree = cKDTree(bh_coords)

    # query the trees for the COPs
    # s_query = s_tree.query_ball_point(cop, r=0.03)
    # g_query = g_tree.query_ball_point(cop, r=0.03)
    # bh_query = g_tree.query_ball_point(cop, r=0.03)

    # OR THIS:
    
    # build kd tree from COPs
    print('building kd tree from COPs...')
    t0 = time.time()
    tree = cKDTree(cop)
    t1 = time.time()
    print(f'that took {t1-t0} s')

    # query it for stellar, gas, black hole particles
    print('querying for stellar particles...')
    t0 = time.time()
    s_query = tree.query_ball_point(s_coords, r=0.03, p=2)
    t1 = time.time()
    print(f'that took {t1-t0} s')
    print('querying for gas particles...')
    t0 = time.time()
    g_query = tree.query_ball_point(g_coords, r=0.03, p=2)
    t1 = time.time()
    print(f'that took {t1-t0} s')
    print('querying for black hole particles...')
    t0 = time.time()
    bh_query = tree.query_ball_point(bh_coords, r=0.03, p=2)
    t1 = time.time()
    print(f'that took {t1-t0} s')

    # collect particles in each galaxy
    n = len(cop)
    print(f'there are {n} galaxies to group particles into')
    s_nbours = {g: [] for g in range(n)}
    g_nbours = {g: [] for g in range(n)}
    bh_nbours = {g: [] for g in range(n)}

    # I don't think we need the ids after all?
    # for s_ind, parts in zip(s_ids, s_query):
    #     for _ind in parts:
    #         s_nbours[_ind].append(s_ind)        
    # for g_ind, parts in zip(g_ids, g_query):
    #     for _ind in parts:
    #         g_nbours[_ind].append(g_ind)
    # for bh_ind, parts in zip(bh_ids, bh_query):
    #     for _ind in parts:
    #         bh_nbours[_ind].append(bh_ind)

    print('grouping stellar particles...')
    t0 = time.time()
    for s_ind, parts in enumerate(s_query):
        for _ind in parts:
            s_nbours[_ind].append(s_ind)
    t1 = time.time()
    print(f'that took {t1-t0} s')

    print('grouping gas particles...')
    t0 = time.time()
    for g_ind, parts in enumerate(g_query):
        for _ind in parts:
            g_nbours[_ind].append(g_ind)
    t1 = time.time()
    print(f'that took {t1-t0} s')
            
    print('grouping black hole particles...')
    t0 = time.time()
    for bh_ind, parts in enumerate(bh_query):
        for _ind in parts:
            bh_nbours[_ind].append(bh_ind)
    t1 = time.time()
    print(f'that took {t1-t0} s')
            
    # let's store these particle groups
    with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc_stellar.pkl', 'wb') as f:
        pickle.dump(s_nbours, f)
    with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc_gas.pkl', 'wb') as f:
        pickle.dump(g_nbours, f)
    with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc_blackhole.pkl', 'wb') as f:
        pickle.dump(bh_nbours, f)

    print('done for this round!')
