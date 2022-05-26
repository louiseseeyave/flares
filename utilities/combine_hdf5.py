"""
Combine HDF5 output files. Assumes the same structure in each file.

    python combine_hdf5.py input_directory tag filetype output_directory

example call: 

    python combine_hdf5.py /cosma7/data/dp004/dc-love2/data/FLARES_DMO/G-EAGLE_39/data/ 001_z014p000 FOF .

"""

import sys
import os
import glob
import re

import numpy as np
import h5py




def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def _get_files_make_dir(_dir, _outdir, tag, dirtype='groups', fname='group_tab'):
    _temp_dir = f"{_outdir}/groups_{tag}" 
    if not os.path.isdir(_temp_dir): os.mkdir(_temp_dir)
    outfile = f"{dirtype}_{tag}/{fname}_{tag}"
    return glob.glob(f"{_dir}/{outfile}*.hdf5"), f"{outfile}.0.hdf5"


def get_files_make_dir(directory, outdir, tag, fileType): 

    if fileType in ['FOF', 'FOF_PARTICLES']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='groups', 
                                            fname='group_tab')
    elif fileType in ['SNIP_FOF', 'SNIP_FOF_PARTICLES']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='groups_snip', 
                                            fname='group_snip_tab')
    elif fileType in ['SUBFIND', 'SUBFIND_GROUP', 'SUBFIND_IDS']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='groups', 
                                            fname='eagle_subfind_tab')
    elif fileType in ['SNIP_SUBFIND', 'SNIP_SUBFIND_GROUP', 'SNIP_SUBFIND_IDS']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='groups_snip', 
                                            fname='eagle_subfind_snip_tab')
    elif fileType in ['SNIP','SNIPSHOT']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='snipshot', 
                                            fname='snip')
    elif fileType in ['SNAP','SNAPSHOT']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='snapshot', 
                                            fname='snap')
    elif fileType in ['PARTDATA']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='particledata', 
                                            fname='eagle_subfind_particles')
    elif fileType in ['SNIP_PARTDATA']:
        files, outfile = _get_files_make_dir(directory, outdir, tag, dirtype='particledata_snip', 
                                            fname='eagle_subfind_snip_particles')
    else:
        raise ValueError("Type of files not supported")
    
    infiles = sorted(files, key=lambda x: int(re.findall("(\d+)", x)[-2]))

    if os.path.isfile(outfile):
        if query_yes_no("File already exists. Remove?"):
            os.remove(outfile)
        else:
            raise ValueError("File note removed, exiting.")

    return infiles, outfile


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


def combine_files(directory, tag, fileType, outdir):

    infiles, outfile = get_files_make_dir(directory, outdir, tag, fileType)

    print(f"Output file: {outdir}/{outfile}")
    print(f"Input file (0): {infiles[0]}")


    for infile in infiles:
        with h5py.File(f"{outdir}/{outfile}", 'a') as hf_out, h5py.File(infile, 'r') as hf_in:
    
            def visitor_func(name, node): 
                dset = node.name
    
                if isinstance(node, h5py.Dataset): 
                    if dset in hf_out:
                        values = hf_in[dset][:]
                        append_to_dataset(hf_out, values, dset)
                    else:
                        _shape = hf_in[dset].shape
                        hf_out.create_dataset(dset, data=hf_in[dset], compression="gzip", 
                                              chunks=True, maxshape=(None,) + _shape[1:])
                        copy_attributes(hf_in[dset], hf_out[dset])
                else: 
                    hf_out.require_group(dset)
                    copy_attributes(hf_in[dset], hf_out[dset])
    
             
            hf_in.visititems(visitor_func)  


if __name__ == "__main__":

    directory = sys.argv[1] 
    tag = sys.argv[2] 
    fileType = sys.argv[3]
    outdir = sys.argv[4]
    
    combine_files(directory, tag, fileType, outdir)


