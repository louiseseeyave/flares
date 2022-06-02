import gc, sys, timeit

import numpy as np
import h5py
from astropy import units as u
from astropy import constants as const

import eagle_IO.eagle_IO as E
import flares

h=0.6777

if __name__ == "__main__":

    ii, tag, inp, data_folder, inpfile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    overwrite=True
    num = str(ii)
    tag = str(tag)
    inp = str(inp)
    data_folder = str(data_folder)
    inpfile = str(inpfile)

    print("Wrting out required properties to hdf5")

    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = F'./{data_folder}/FLARES_{num}_sp_info.hdf5'
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9' or 'RECAL' in inp:
        filename = F'./{data_folder}/EAGLE_{inp}_sp_info.hdf5'
        sim_type = 'PERIODIC'

    else:
        ValueError("Type of input simulation not recognized")

    fl = flares.flares(fname = filename,sim_type = sim_type)
    fl.create_group(tag)
    if inp == 'FLARES':
        dir = fl.directory
        sim = F"{dir}GEAGLE_{num}/data/"

    elif inp == 'REF':
        sim = fl.ref_directory

    elif inp == 'AGNdT9':
        sim = fl.agn_directory

    elif 'RECAL' in inp:
        sim = vars(fl)[F'{inp.lower()}_directory']

    else:
        ValueError("Type of input simulation not recognized")


    with h5py.File(filename, 'r') as hf:
        ok_centrals = np.array(hf[tag+'/Galaxy'].get('Central_Indices'), dtype = np.int64)
        indices     = np.array(hf[tag+'/Galaxy'].get('Indices'), dtype = np.int64)
        dindex      = np.array(hf[tag+'/Particle'].get('DM_Index'), dtype = np.int64)
        sindex      = np.array(hf[tag+'/Particle'].get('S_Index'), dtype = np.int64)
        gindex      = np.array(hf[tag+'/Particle'].get('G_Index'), dtype = np.int64)
        bhindex     = np.array(hf[tag+'/Particle'].get('BH_Index'), dtype = np.int64)

    nThreads=8
    a = E.read_header('SUBFIND', sim, tag, 'ExpansionFactor')
    z = E.read_header('SUBFIND', sim, tag, 'Redshift')

    data = np.genfromtxt(inpfile, delimiter=',', dtype='str')

    for ii in range(len(data)):
        name = data[:,0][ii]
        path = data[:,1][ii]
        unit = data[:,3][ii]
        desc = data[:,4][ii]
        CGS  = data[:,5][ii]

        if 'PartType' in path:
            tmp = 'PARTDATA'
            location = 'Particle'
            if 'PartType0' in path:
                sel = gindex
            elif 'PartType1' in path:
                sel = dindex
            elif 'PartType4' in path:
                sel = sindex
            elif 'PartType5' in path:
                sel = bhindex
        else:
            tmp = 'SUBFIND'
            location = 'Galaxy'
            if 'FOF' in path:
                sel = ok_centrals
            else:
                sel = indices

        sel = np.asarray(sel, dtype=np.int64)
        try:
            out = E.read_array(tmp, sim, tag, path, noH=True, physicalUnits=True, numThreads=nThreads, CGS=eval(CGS))[sel]
        except:
            print("read_array failed")

            if 'coordinates' in path.lower():
                out = np.zeros((len(indices),3))
            elif 'velocity' in path.lower():
                out = np.zeros((len(indices),3))
            elif 'halfmassrad' in path.lower():
                out = np.zeros((len(indices),6))
            else:
                out = np.zeros(len(indices))


        if 'age' in name.lower(): out = fl.get_age(out, z, nThreads)
        # if 'PartType5' in path:
        #     if len(out.shape)>1:
        #         out[nok] = [0.,0.,0.]
        #     else:
        #         out[nok] = 0.


        if 'coordinates' in path.lower(): out = out.T/a
        if 'velocity' in path.lower(): out = out.T
        # if 'halfmassrad' in path.lower(): out = out.T
        if name=='BH_Mdot':
            out = h*(out.astype(np.float64)*(u.g/u.s)).to(u.M_sun/u.yr).value


        fl.create_dataset(out, name, '{}/{}'.format(tag, location),
                          desc = desc.encode('utf-8'), unit = unit.encode('utf-8'),
                          overwrite=overwrite)

        del out

    print (F'Completed writing required datasets from {inpfile}')
