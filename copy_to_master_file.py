import sys
import pandas as pd
import h5py
import flares

df = pd.read_csv('weight_files/weights_grid.txt')
df.drop(columns=['Unnamed: 0'], inplace=True)

fl = flares.flares('./data/flares.hdf5',sim_type='FLARES')
in_dir = './data/'

"""
Datasets should include full path after halo/tag/
Pass as many to the command line as you wish to copy
"""
dsets = sys.argv[1:]
for dset in dsets:
    print(dset)

with h5py.File('./data/flares.hdf5','a') as outfile:
    for ii, halo in enumerate(fl.halos):
        print(halo)
        # fl.create_group(halo)

        # hdr = df.iloc[ii:ii+1].to_dict('list')
        # for key, value in hdr.items():
        #     outfile[halo].attrs[key] = value[0]

        infile = h5py.File('%s/FLARES_%s_sp_info.hdf5'%(in_dir,halo),'r')

        for tag in fl.tags:
            for dset in dsets:
                # infile.copy(f'{tag}/{dset}', outfile[halo])

                if f'{halo}/{tag}/{dset}' not in outfile:
                    group_path = infile[f'{tag}/{dset}'].parent.name
                    group_id = outfile.require_group(group_path)
                    infile.copy(f'{tag}/{dset}', outfile[f'{halo}/{group_path}'])
                else:
                    # ---- if not the same size, this will fail 
                    # ---- (and you're probably doing something you shouldn't)
                    # ---- can force past by using code below
                    # data = outfile[f'{halo}/{tag}/{dset}']
                    # data[...] = infile[f'{tag}/{dset}'][:]
                    
                    del outfile[f'{halo}/{tag}/{dset}']
                    outfile.create_dataset(f'{halo}/{tag}/{dset}', 
                                           data=infile[f'{tag}/{dset}'][:])

        infile.close()
