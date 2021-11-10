import numpy as np
import pandas as pd
#from mpi4py import MPI
import h5py
import re
import sys
from functools import partial
import schwimmbad
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import flares
from synthobs.sed import models
from phot_modules import get_lum, get_flux, get_lines, get_SED
import flare.filters


def get_simtype(inp):

    if inp == 'FLARES':
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9':
        sim_type = 'PERIODIC'

    return sim_type

def lum_write_out(num, tag, kappa, BC_fac, filters = flare.filters.TH[:-1], inp = 'FLARES', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='default'):


    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = F"./{data_folder}/FLARES_{num}_sp_info.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num = '00'

    else:
        ValueError(F"No input option of {inp}")


    lumintr = get_lum(num, 0, tag, BC_fac, filters = filters, LF = False, inp = inp, Type = 'Intrinsic', extinction = extinction, data_folder = data_folder, aperture = aperture)
    lumstell = get_lum(num, 0, tag, BC_fac, filters = filters, LF = False, inp = inp, Type = 'Pure-stellar', extinction = extinction, data_folder = data_folder, aperture = aperture)
    lumBC = get_lum(num, 0, tag, BC_fac, filters = filters, LF = False, inp = inp, Type = 'Only-BC', extinction = extinction, data_folder = data_folder, aperture = aperture)
    lumatt = get_lum(num, kappa, tag, BC_fac, filters = filters, LF = False, inp = inp, Type = 'Total', log10t_BC =  log10t_BC, extinction = extinction, data_folder = data_folder, aperture = aperture)

    # print("Output array dimensions:",lumintr.shape, lumstell.shape, lumBC.shape, lumatt.shape)

    # print(lumintr)

    fl = flares.flares(fname = filename, sim_type = sim_type)

    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Pure_Stellar")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/No_ISM")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI")

    for ii, jj in enumerate(filters):

        _filter = jj[8:]

        fl.create_dataset(values = lumintr[:,ii], name = F"{_filter}",
                          group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic",
                          desc = F'Intrinsic (stellar + nebular) luminosity of the galaxy in the {_filter} band', unit = "ergs/s/Hz", overwrite=True)

        fl.create_dataset(values = lumstell[:,ii], name = F"{_filter}",
                          group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Pure_Stellar",
                          desc = F'Stellar luminosity of the galaxy in the {_filter} band', unit = "ergs/s/Hz", overwrite=True)

        fl.create_dataset(values = lumBC[:,ii], name = F"{_filter}",
                          group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/No_ISM",
                          desc = F'Intrinsic (stellar + nebular) luminosity of the galaxy with BC attenuation in the {_filter} band with a birth cloud factor of {BC_fac} following {extinction} curve',
                          unit = "ergs/s/Hz", overwrite=True)

        fl.create_dataset(values = lumatt[:,ii], name = F"{_filter}",
                          group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI",
                          desc = F"Dust corrected luminosity (using ModelI) of the galaxy in the {_filter} band with a birth cloud factor of {BC_fac} following {extinction} curve",
                          unit = "ergs/s/Hz", overwrite=True)


def flux_write_out(num, tag, kappa, BC_fac, filters = flare.filters.ACS, inp = 'FLARES', extinction = 'default', data_folder = 'data', aperture='default'):

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = F"./{data_folder}/FLARES_{num}_sp_info.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    else:
        ValueError(F"No input option of {inp}")

    fluxintr = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp, Type = 'Intrinsic', extinction = extinction, data_folder = data_folder, aperture = aperture)
    fluxstell = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp, Type = 'Pure-stellar', extinction = extinction, data_folder = data_folder, aperture = aperture)
    fluxBC = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp, Type = 'Only-BC', extinction = extinction, data_folder = data_folder, aperture = aperture)
    fluxatt = get_flux(num, kappa, tag, BC_fac, filters = filters, inp = inp, Type = 'Total', extinction = extinction, data_folder = data_folder, aperture = aperture)

    fl = flares.flares(fname = filename, sim_type = sim_type)
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Intrinsic")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Pure_Stellar")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/No_ISM")
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/DustModelI")

    for ii, jj in enumerate(filters):

        _filter = re.findall('\w+', jj) #index `0` is the telescope, `1` is the name
        #of the instrument and `2` is the name of the filter

        fl.create_dataset(values = fluxintr[:,ii], name = F"{_filter[0]}/{_filter[1]}/{_filter[2]}",
        group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Intrinsic",
        desc = F"Intrinsic (stellar + nebular) flux of the galaxy in the {_filter}", unit = "nJy", overwrite=True)

        fl.create_dataset(values = fluxstell[:,ii], name = F"{_filter[0]}/{_filter[1]}/{_filter[2]}",
        group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/Pure_Stellar",
        desc = F"Stellar flux of the galaxy in the {_filter}", unit = "nJy", overwrite=True)

        fl.create_dataset(values = fluxBC[:,ii], name = F"{_filter[0]}/{_filter[1]}/{_filter[2]}",
        group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/No_ISM",
        desc = F"Intrinsic (stellar + nebular) flux of the galaxy with BC attenuation in the {_filter} with a birth cloud factor of {BC_fac} following {extinction} curve", unit = "nJy", overwrite=True)

        fl.create_dataset(values = fluxatt[:,ii], name = F"{_filter[0]}/{_filter[1]}/{_filter[2]}",
        group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Flux/DustModelI",
        desc = F"Dust corrected flux (using ModelI) of the galaxy in the {_filter} with a birth cloud factor of {BC_fac} following {extinction} curve", unit = "nJy", overwrite=True)


def line_write_out(num, lines, tag, kappa, BC_fac, label, inp = 'FLARES', LF = False, log10t_BC = 7., Type = 'Total', extinction = 'default', data_folder = 'data', aperture='default'):

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = F"./{data_folder}/FLARES_{num}_sp_info.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num='00'

    else:
        ValueError(F"No input option of {inp}")

    calc = partial(get_lines, sim=num, kappa = kappa, tag = tag, BC_fac = BC_fac, inp = inp, IMF = 'Chabrier_300', LF = False, log10t_BC = log10t_BC, Type = Type, extinction = extinction, data_folder = data_folder, aperture = aperture)

    pool = schwimmbad.MultiPool(processes=8)
    dat = np.array(list(pool.map(calc, lines)))
    pool.close()

    for ii, line in enumerate(lines):
        out_lum = dat[:,0][ii]
        out_EW = dat[:,1][ii]


        if Type == 'Total':
            fl = flares.flares(fname = filename, sim_type = sim_type)
            fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI", verbose=True)
            print (F'{line} is being written to disk')
            fl.create_dataset(values = out_lum, name = F"{line}/Luminosity",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI",
            desc = F"Dust corrected luminosity (using ModelI) of the galaxy with a birth cloud factor of {BC_fac} following {extinction} curve", unit = "ergs/s", overwrite=True)

            fl.create_dataset(values = out_EW, name = F"{line}/EW",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI",
            desc = F"EW (using ModelI) of the galaxy with a birth cloud factor of {BC_fac} following {extinction} curve", unit = "Angstrom", overwrite=True)

        elif Type=='Intrinsic':
            fl = flares.flares(fname = filename, sim_type = sim_type)
            fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}", verbose=True)
            print (F'{line} is being written to disk')
            fl.create_dataset(values = out_lum, name = F"{line}/Luminosity",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}",
            desc = F"Intrinsic line luminosity", unit = "ergs/s", overwrite=True)

            fl.create_dataset(values = out_EW, name = F"{line}/EW",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}",
            desc = F"Intrinsic EW", unit = "Angstrom", overwrite=True)

        elif Type=='Only-BC':
            fl = flares.flares(fname = filename, sim_type = sim_type)
            fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}", verbose=True)
            print (F'{line} is being written to disk')
            fl.create_dataset(values = out_lum, name = F"{line}/Luminosity",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}",
            desc = F"Intrinsic line luminosity with birth cloud factor {BC_fac} following {extinction} curve", unit = "ergs/s", overwrite=True)

            fl.create_dataset(values = out_EW, name = F"{line}/EW",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/{label}",
            desc = F"Intrinsic EW with birth cloud factor {BC_fac} following {extinction} curve", unit = "Angstrom", overwrite=True)


def sed_write_out(num, tag, kappa, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300', log10t_BC = 7., extinction = 'default', data_folder = 'data', aperture='default'):

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = F"./{data_folder}/FLARES_{num}_sp_info.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num='00'

    else:
        ValueError(F"No input option of {inp}")

    try:
        dat     = get_SED(num, kappa, tag, BC_fac, inp=inp, log10t_BC=log10t_BC, extinction=extinction, data_folder=data_folder, aperture=aperture)
        lam     = dat[0][0]
        stell   = dat[1]
        intr    = dat[2]
        noISM   = dat[3]
        dust    = dat[4]
    except:
        lam     = np.array([])
        stell   = np.array([])
        intr    = np.array([])
        noISM   = np.array([])
        dust    = np.array([])

    fl = flares.flares(fname = filename, sim_type = sim_type)
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED")


    fl.create_dataset(values = lam, name = F"Wavelength",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"Wavelength array for the SED", unit = "Angstrom", overwrite=True)

    fl.create_dataset(values = stell, name = F"Pure_Stellar",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"Pure stellar SED", unit = "ergs/s/Hz", overwrite=True)

    fl.create_dataset(values = intr, name = F"Intrinsic",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"Intrinisc SED", unit = "ergs/s/Hz", overwrite=True)

    fl.create_dataset(values = noISM, name = F"No_ISM",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"SED from attenuation only from the birth cloud component with kappa_BC={BC_fac} and {extinction} curve", unit = "ergs/s/Hz", overwrite=True)

    fl.create_dataset(values = dust, name = F"DustModelI",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"SED from DustModelI with kappa_BC={BC_fac} and {extinction} curve", unit = "ergs/s/Hz", overwrite=True)




if __name__ == "__main__":

    #Filters for flux: ACS + WFC3NIR_W, IRAC, Euclid, Subaru, NIRCam

    # BC_facs =     [0.,0.001, 0.1,  0.25, 0.5, 0.75, 1.,  1.25,  1.5,  1.75,    2.]
    # kappas =  [1.018, 1.016, 0.953, 0.839, 0.234, 0.103, 0.0795, 0.0609, 0.0485, 0.0407, 0.037]
    # extinction = ['SMC', 'N18']
    # kappas = [0.0691, 0.22]

    ii, tag, inp, prop, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    num = str(ii)
    tag = str(tag)
    data_folder = str(data_folder)

    BC_fac = 1.
    kappa = 0.0795
    sim_type = get_simtype(inp)
    fl = flares.flares(fname = 'tmp', sim_type = sim_type)
    tags = fl.tags
    print (num, tag, inp, data_folder)

    if prop == 'Luminosity':
        print ("Calculating luminosities")
        lum_write_out(num=num, tag = tag, kappa = kappa, BC_fac = BC_fac, inp = inp, data_folder = data_folder, filters=flare.filters.TH[:-1])

    elif prop == 'Flux':
        print ("Calculating fluxes")
        flux_write_out(num=num, tag = tag, kappa = kappa, BC_fac = BC_fac,
                       filters = flare.filters.ACS + flare.filters.WFC3NIR_W + flare.filters.IRAC + flare.filters.Euclid + flare.filters.Subaru + flare.filters.NIRCam + flare.filters.MIRI,
                       inp = inp, data_folder = data_folder)

    elif prop == 'Lines':
        print ("Calculating line luminosities and EW")
        m=models.EmissionLines("BPASSv2.2.1.binary/Chabrier_300", verbose=False)
        lines = m.lines

        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Total', inp = inp, LF = False, log10t_BC = 7., label = 'Total', data_folder = data_folder)
        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Intrinsic', inp = inp, LF = False, log10t_BC = 7., label = 'Intrinsic', data_folder = data_folder)
        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Only-BC', inp = inp, LF = False, log10t_BC = 7., label = 'No_ISM', data_folder = data_folder)

    elif prop == 'SED':
        print ("Calculating SEDs")
        sed_write_out(num, tag, kappa, BC_fac, inp = inp, IMF = 'Chabrier_300', log10t_BC = 7., data_folder = data_folder)

    else:
        ValueError(F"No input of type {prop}")
