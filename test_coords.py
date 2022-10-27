import numpy as np
import eagle_IO.eagle_IO as E
import h5py
import pickle

rg = '00'
snap = '010_z005p000'

# in the master file documentation, the units are stated to be in cMpc
master = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
with h5py.File(master, 'r') as m:
    s_coords = np.array(m[f'{rg}/{snap}/Particle/S_Coordinates'])
    units = m[f'{rg}/{snap}/Particle/S_Coordinates'].attrs['Units']
    print('the s coord units in the master file are stated to be:', units)
    s_coords = np.transpose(s_coords)
    print('the first set of s coords is:', s_coords[0])
    print('after converting to pMpc, it is:', s_coords[0]/(1+5))
    
    cop = np.array(m[f'{rg}/{snap}/Galaxy/COP'])
    units = m[f'{rg}/{snap}/Galaxy/COP'].attrs['Units']
    print('the COP units in the master file are stated to be:', units)
    cop = np.transpose(cop)
    print('the first set of cop coords is:', cop[0])

# eagle io reads in the coordinats in pMpc
nThreads = 8
sim = f'/cosma7/data/dp004/FLARES/FLARES-1/flares_{rg}/data'
s_coords = E.read_array('PARTDATA', sim, snap, '/PartType4/Coordinates',
                        noH=True, physicalUnits=True, numThreads=nThreads,
                        CGS=False)
print('the first few sets of s coords are:', s_coords[0:5])

# now let's check the files in data_valid_particles
with open('data_valid_particles/flares_39_011_z004p770_30pkpc_stellar.pkl', 'rb') as f:
    ff = pickle.load(f)
print(ff)

# phew, it works.
