import numpy as np
import h5py

regions = np.array(['00','01','02','03','04','05','06','07','08','09',
                    '10','11','12','13','14','15','16','17','18','19',
                    '20','21','22','23','24','25','26','27','28','29',
                    '30','31','32','33','34','35','36','37','38','39'])

snaps = np.array(['000_z015p000','001_z014p000','002_z013p000','003_z012p000',
                  '004_z011p000','005_z010p000','006_z009p000','007_z008p000',
                  '008_z007p000','009_z006p000','010_z005p000','011_z004p770'])

# paths to the dataset that you want to copy to the master file:
# paths = [('Galaxy','IonisingEmissivity'), ('Galaxy','IonisingPPE'),
#          ('Galaxy','IonisationParameter'), ('Galaxy/Metallicity','MassWeightedYoungStellarZ')]

paths = [ ('Galaxy','IonisationParameter')]

# master file that data will be added to:
# master = '/cosma7/data/dp004/dc-seey1/data/flares/scripts/flares.hdf5'
# master = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'
master = '/cosma7/data/dp004/dc-seey1/data/flares/steve/flares.hdf5'

m = h5py.File(master, 'a')

for rg in regions:

    # file that is being read
    hf = h5py.File(f'FLARES_{rg}_sp_info.hdf5', 'r')

    for snap in snaps:
        for (path, name) in paths:

            print(f'region {rg}, snap {snap}, quantity {name}')

            if f'{rg}/{snap}/{path}/{name}' in m:
                # print('something is wrong')
                print('deleting pre-existing dataset before adding')
                del m[f'{rg}/{snap}/{path}/{name}']
                m.create_dataset(f'{rg}/{snap}/{path}/{name}', data=hf[f'{snap}/{path}/{name}'][:])
            else:
                hf.copy(f'{snap}/{path}/{name}', m[f'{rg}/{snap}/{path}'])
                
print('done :)')
