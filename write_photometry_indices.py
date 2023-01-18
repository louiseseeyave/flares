import h5py, sys
import numpy as np

import flares


if __name__ == "__main__":

    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    num = str(ii)
    tag = str(tag)
    data_folder = str(data_folder)

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = F"./{data_folder}/FLARES_{num}_sp_info.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9') or ('RECAL' in inp):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num = '00'

    else:
        ValueError(F"No input option of {inp}")

    add_D4000 = True
    add_BB = True
    add_bolometric = True
    add_UVC = True
    add_BB_Wilkins = True
    add_BB_Binggeli = True   #https://arxiv.org/abs/1908.11393 Bingelli+2019


    fl = flares.flares(filename, sim_type=sim_type)

    print (F"Running for {tag} in region {num}")

    # create indices groups if it doesn't already exist

    model_tag = f'{tag}/Galaxy/BPASS_2.2.1/Chabrier300'

    fl.create_group(f'{model_tag}/Indices')



    # -----------------------------------------------
    # D4000
    if add_D4000:

        indice_tag = f'{model_tag}/Indices/D4000'

        fl.create_group(indice_tag)

        for spec_type in ['DustModelI','Intrinsic','No_ISM','Pure_Stellar']:
            try:
                with h5py.File(filename, mode='r') as hf:
                    lam = np.array(hf[f'{model_tag}/SED/Wavelength'][:])
                    s1 = ((lam>=3750)&(lam<3950)).nonzero()[0]
                    s2 = ((lam>=4050)&(lam<4250)).nonzero()[0]

                    fnu = np.array(hf[f'{model_tag}/SED/{spec_type}'][:])
                    tmp = np.sum(fnu[:, s2], axis=1)/np.sum(fnu[:, s1], axis=1)
            except:
                tmp = np.array([])

            fl.create_dataset(values = tmp, name = F"{spec_type}",
            group = F"{indice_tag}",
            desc = F'D4000, ratio of flux between 4050-4250 and 3750-3950', unit = "No unit", overwrite=True)


    # -----------------------------------------------
    # 3400 - 3600 # 4150 - 4250
    if add_BB_Wilkins:

        indice_tag = f'{model_tag}/Indices/BB_Wilkins'

        fl.create_group(indice_tag)


        for spec_type in ['DustModelI','Intrinsic','No_ISM','Pure_Stellar']:
            try:
                with h5py.File(filename, mode='r') as hf:
                    lam = np.array(hf[f'{model_tag}/SED/Wavelength'][:])

                    s1 = ((lam>=3400)&(lam<3600)).nonzero()[0]
                    s2 = ((lam>=4150)&(lam<4250)).nonzero()[0]
                    fnu = np.array(hf[f'{model_tag}/SED/{spec_type}'][:])
                    tmp = (np.sum(fnu[:, s2], axis=1)/np.sum(fnu[:, s1], axis=1))/(len(lam[s2])/len(lam[s1]))
            except:
                tmp = np.array([])

            fl.create_dataset(values = tmp, name = F"{spec_type}",
            group = F"{indice_tag}",
            desc = F'Balmer break as defined by Steve Wilkins, ratio of flux between 4150-4250 and 3400-3600', unit = "No unit", overwrite=True)


    # -----------------------------------------------
    # Balmer Break using method by Binggeli (https://arxiv.org/pdf/1908.11393.pdf). Just the flux at 4200 / 3500.
    if add_BB_Binggeli:

        indice_tag = f'{model_tag}/Indices/BB_Binggeli'

        fl.create_group(indice_tag)

        for spec_type in ['DustModelI','Intrinsic','No_ISM','Pure_Stellar']:
            try:
                with h5py.File(filename, mode='r') as hf:
                    lam = np.array(hf[f'{model_tag}/SED/Wavelength'][:])
                    fnu = np.array(hf[f'{model_tag}/SED/{spec_type}'][:])
                    tmp = fnu[:, 4199]/fnu[:, 3499] # I believe this should be correct since d\lam = 1\AA
            except:
                tmp = np.array([])

            fl.create_dataset(values = tmp, name = F"{spec_type}",
            group = F"{indice_tag}",
            desc = F'Balmer break as defined by Bingelli+2019, ratio of flux at 4199 and 3499', unit = "No unit", overwrite=True)




    # -----------------------------------------------
    # Bolometric Luminosity
    if add_bolometric:

        indice_tag = f'{model_tag}/Indices/Lbol'

        fl.create_group(indice_tag)

        for spec_type in ['DustModelI','Intrinsic','No_ISM','Pure_Stellar']:
            try:
                with h5py.File(filename, mode='r') as hf:
                    lam = np.array(hf[f'{model_tag}/SED/Wavelength'][:])
                    fnu = np.array(hf[f'{model_tag}/SED/{spec_type}'][:])
                    flam = fnu*3E8/(lam*1E-10)**2
                    tmp = np.sum(flam, axis=1)*1E-10
            except:
                tmp = np.array([])

            fl.create_dataset(values = tmp, name = F"{spec_type}",
            group = F"{indice_tag}",
            desc = F'Bolometric luminosity', unit = "ergs/s", overwrite=True)


    # -----------------------------------------------
    # UV continuum slope (Beta)

    if add_UVC:

        indice_tag = f'{model_tag}/Indices/beta'

        fl.create_group(indice_tag)

        for spec_type in ['DustModelI','Intrinsic','No_ISM','Pure_Stellar']:
            try:
                with h5py.File(filename, mode='r') as hf:
                    FUV = np.array(hf[f'{model_tag}/Luminosity/{spec_type}/FUV'][:])
                    NUV = np.array(hf[f'{model_tag}/Luminosity/{spec_type}/NUV'][:])
                    beta = (np.log10(FUV/NUV)/np.log10(1500/2500))-2.0
                    tmp = beta
            except:
                tmp = np.array([])

            fl.create_dataset(values = tmp, name = F"{spec_type}",
            group = F"{indice_tag}",
            desc = F'UV continuum slope', unit = "No unit", overwrite=True)
