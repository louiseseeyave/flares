import h5py
import flares

df = pd.read_csv('weight_files/weights_grid.txt')
df.drop(columns=['Unnamed: 0'], inplace=True)

fl = flares.flares('./data/flares.hdf5',sim_type='FLARES')

in_dir = './data/'

with h5py.File('./data/flares.hdf5','a') as outfile:

    for halo in fl.halos:
        print(halo)

        fl.create_group(F'{halo}')

        hdr = df.iloc[ii:ii+1].to_dict('list')
        fl.create_header(F'{halo}/Header', hdr)

        infile = h5py.File('%s/FLARES_%s_sp_info.hdf5'%(in_dir,halo),'r')

        for tag in fl.tags:
            infile.copy(tag,outfile[halo])

        infile.close()
