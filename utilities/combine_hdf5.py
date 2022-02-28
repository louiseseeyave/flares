import numpy as np
import h5py
import sys

import eagle_IO.eagle_IO as E

directory = sys.argv[1] # '/cosma7/data/dp004/dc-love2/data/FLARES_DMO/G-EAGLE_39/data/'
tag = sys.argv[2] # '001_z014p000' 
filetype = sys.argv[3] # 'FOF' 
outfile = sys.argv[4] # 'group_tab_001_z014p000.0.hdf5'

infiles = E.get_files(filetype, directory, tag)


def copy_attributes(in_object, out_object):
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value


def append_to_dataset(out_hf, values, dataset):
    dset = out_hf[dataset]
    ini = len(dset)

    if np.isscalar(values):
        add = 1
    else:
        add = len(values)

    dset.resize(ini+add, axis = 0)
    dset[ini:] = values
    out_hf.flush()



for infile in infiles:
    print(infile,outfile)
    with h5py.File(outfile, 'a') as hf_out, h5py.File(infile, 'r') as hf_in:

        def visitor_func(name, node): 
            dset = node.name

            if isinstance(node, h5py.Dataset): 
                if dset in hf_out:
                    # print(dset, "appending")
                    values = hf_in[dset][:]
                    append_to_dataset(hf_out, values, dset)
                else:
                    # print(dset, "creating")
                    _shape = hf_in[dset].shape
                    hf_out.create_dataset(dset, data=hf_in[dset], compression="gzip", 
                                          chunks=True, maxshape=(None,) + _shape[1:])
                    copy_attributes(hf_in[dset], hf_out[dset])
            else: 
                hf_out.require_group(dset)
                copy_attributes(hf_in[dset], hf_out[dset])

         
        hf_in.visititems(visitor_func)  

