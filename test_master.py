

import h5py


# f = '/cosma7/data/dp004/dc-love2/codes/flares/data/flares.hdf5'
f = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'

# with h5py.File(f, mode='r') as hf:
#
#     sims = hf.keys()
#     tags = hf[list(sims)[0]].keys()
#
#     for sim in sims:
#         for tag in tags:
#             Datasetnames=hf[sim][tag].keys()
#             print(sim, tag, Datasetnames)


with h5py.File(f, mode='r') as hf:

    sims = hf.keys()
    tags = hf[list(sims)[0]].keys()

    for sim in sims:
        for tag in tags:

            try:
                Ngal_Mstar = len(hf[sim][tag]['Galaxy/Mstar'][:])
            except:
                Ngal_Mstar = 'FAILED'

            try:
                Ngal_Lum = len(hf[sim][tag]['Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/FUV'][:])
            except:
                Ngal_Lum = 'FAILED'

            if Ngal_Lum == 'FAILED' or Ngal_Mstar == 'FAILED':
                print(sim, tag, Ngal_Mstar, Ngal_Lum)
            elif Ngal_Mstar-Ngal_Lum != 0:
                print(sim, tag, Ngal_Mstar, Ngal_Lum)



            # else:
            #     print(sim, tag, Ngal_Mstar, Ngal_Lum, Ngal_Mstar-Ngal_Lum)
