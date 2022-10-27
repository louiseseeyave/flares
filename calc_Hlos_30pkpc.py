import sys
import numpy as np
import eagle_IO.eagle_IO as E
import pickle
from calc_Hlos import new_cal_HLOS
import h5py

from astropy import units as u
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

# calculate H los for all stellar particles within 30 pkpc of COP

if __name__ == "__main__":

    ii, snap = sys.argv[1], sys.argv[2]

    # test:
    # ii, snap = 0, '010_z005p000'

    if len(str(ii)) == 1:
        rg = '0' + str(ii)
    else:
        rg = str(ii)

    print(f'region {rg}, snapshot {snap}')

    # LET'S CHECK OUR DATA -------------------------------------------

    # get number of galaxies in master file
    master = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
    with h5py.File(master, 'r') as m:
        raw_index = np.array(m[f'{rg}/{snap}/Galaxy/Indices'])
    print('number of galaxies in master:', len(raw_index))

    # check that there is the same number in the particle neighbour files
    with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc.pkl', 'rb') as f:
        nbours = pickle.load(f)
        s_nbours = nbours['stellar_neighbours']
        g_nbours = nbours['gas_neighbours']        

    if len(s_nbours)==len(raw_index):
        print('stellar particle file has', len(s_nbours), 'galaxies')
    else:
        print(f"the number of galaxies {len(s_nbours)} doesn't match up! (stellar)")
        exit()

    if len(g_nbours)==len(raw_index):
        print('gas particle file has', len(g_nbours), 'galaxies')
    else:
        print(f"the number of galaxies {len(g_nbours)} doesn't match up! (gas)")
        exit()

    # LET'S LOAD SOME DATA -------------------------------------------
    
    # particle coordinates [pMpc]
    nThreads = 8
    sim = f'/cosma7/data/dp004/FLARES/FLARES-1/flares_{rg}/data'
    s_coords = E.read_array('PARTDATA', sim, snap, '/PartType4/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    g_coords = E.read_array('PARTDATA', sim, snap, '/PartType0/Coordinates',
                            noH=True, physicalUnits=True, numThreads=nThreads,
                            CGS=False)
    # bh_coords = E.read_array('PARTDATA', sim, snap, '/PartType5/Coordinates',
    #                         noH=True, physicalUnits=True, numThreads=nThreads,
    #                         CGS=False)
    s_coords = np.array(s_coords, dtype=np.float64)
    g_coords = np.array(g_coords, dtype=np.float64)
    # bh_coords = np.array(bh_coords, dtype=np.float64)
    print('s, g coord shapes:', s_coords.shape, g_coords.shape)

    # gas temp [K]
    g_temp = E.read_array('PARTDATA', sim, snap, '/PartType0/Temperature',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    g_temp = np.array(g_temp, dtype=np.float64)

    # gas particle hydrogen mass fraction
    g_hfrac = E.read_array('PARTDATA', sim, snap, '/PartType0/SmoothedElementAbundance/Hydrogen',
                           noH=True, physicalUnits=True, numThreads=nThreads,
                           CGS=False)
    g_hfrac = np.array(g_hfrac, dtype=np.float64)

    # gas particle mass
    g_mass = E.read_array('PARTDATA', sim, snap, '/PartType0/Mass',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    g_mass = np.array(g_mass, dtype=np.float64)*1e10

    # gas particle smoothing length [pMpc]
    g_sml = E.read_array('PARTDATA', sim, snap, '/PartType0/SmoothingLength',
                          noH=True, physicalUnits=True, numThreads=nThreads,
                          CGS=False)
    g_sml = np.array(g_sml, dtype=np.float64)

    # convert comoving coordinates to physical (no need)
    # if len(raw_index)==0:
    #     print (f"No galaxies in region {rg} for snapshot {snap}")
    # else:
    #     z = float(snap[5:].replace('p','.'))
    #     print(f'redshift {z}, converting to physical coordinates...')
    #     s_coords=s_coords/(1+z)
    #     g_coords=g_coords/(1+z)
    #     bh_coords=bh_coords/(1+z)

    # LET'S GET F_ESC ------------------------------------------------

    # get kernel
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    print('lkernel:', lkernel)
    header = kinp['header']
    kbins = header.item()['bins']

    s_length = []
    g_length = []
    s_hlos = np.array([])

    # print('s_coords:', s_coords[0:5])
    # print('g_coords:', g_coords[0:5])
    # print('g_mass:', g_mass[0:5])
    # print('g_hfrac:', g_hfrac[0:5])
    # print('g_temp:', g_temp[0:5])
    # print('g_sml:', g_sml[0:5])
    
    for ii in range(len(raw_index)): # for each galaxy

        # get masks
        s_keep = s_nbours[ii]
        print(f'there are {len(s_keep)} stellar particles in this galaxy')
        g_keep = g_nbours[ii]
        print(f'there are {len(g_keep)} gas particles in this galaxy')

        # get particles
        hlos = new_cal_HLOS(s_coords[s_keep], g_coords[g_keep], g_mass[g_keep],
                            g_hfrac[g_keep], g_temp[g_keep], g_sml[g_keep],
                            lkernel, kbins)*conv

        s_length.append(len(s_keep))
        g_length.append(len(g_keep))
        s_hlos = np.append(s_hlos, hlos)
    
    newfile = f'/cosma7/data/dp004/dc-seey1/data/flares/temp/flares_{rg}.hdf5'
    print(f'saving data in {newfile}...')
    with h5py.File(newfile, 'a') as fl:

        if f'{rg}/{snap}/Particle/S_Hlos_notsubfind' in fl:
            print(f'deleting {rg}/{snap}/Particle/S_Hlos_notsubfind')
            del fl[f'{rg}/{snap}/Particle/S_Hlos_notsubfind']
        if f'{rg}/{snap}/Galaxy/S_Length_notsubfind' in fl:
            print(f'deleting {rg}/{snap}/Galaxy/S_Length_notsubfind')
            del fl[f'{rg}/{snap}/Galaxy/S_Length_notsubfind']
        if f'{rg}/{snap}/Galaxy/G_Length_notsubfind' in fl:
            print(f'deleting {rg}/{snap}/Galaxy/G_Length_notsubfind')
            del fl[f'{rg}/{snap}/Galaxy/G_Length_notsubfind']
        
        fl.create_dataset(f'{rg}/{snap}/Particle/S_Hlos_notsubfind', data = s_hlos)
        fl[f'{rg}/{snap}/Particle/S_Hlos_notsubfind'].attrs['Description'] = f'Stellar particle line-of-sight H column density along the z-axis for particles within 30 pkpc of COP - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Particle/S_Hlos_notsubfind'].attrs['Units'] = 'Msun/pc^2'

        fl.create_dataset(f'{rg}/{snap}/Galaxy/S_Length_notsubfind', data = s_length)
        fl[f'{rg}/{snap}/Galaxy/S_Length_notsubfind'].attrs['Description'] = f'No. of stellar particles within 30 pkpc of COP - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Galaxy/S_Length_notsubfind'].attrs['Units'] = 'None'

        fl.create_dataset(f'{rg}/{snap}/Galaxy/G_Length_notsubfind', data = g_length)
        fl[f'{rg}/{snap}/Galaxy/G_Length_notsubfind'].attrs['Description'] = f'No. of gas particles within 30 pkpc of COP - NOT SUBFIND!'
        fl[f'{rg}/{snap}/Galaxy/G_Length_notsubfind'].attrs['Units'] = 'None'

    print('done for this round!')
        
        
