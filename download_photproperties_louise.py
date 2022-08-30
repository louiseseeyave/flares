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
import os
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare' # path to Steve's flare folder
import flare.filters


def get_simtype(inp):

    if inp == 'FLARES':
        sim_type = 'FLARES'

    elif inp == 'REF' or inp == 'AGNdT9':
        sim_type = 'PERIODIC'

    return sim_type


def lum_write_out(num, tag, kappa, BC_fac, filters = flare.filters.TH[:-1],
                  inp = 'FLARES', log10t_BC = 7., extinction = 'default',
                  data_folder = 'data', aperture='30', fescs = None):

    """
    add luminosity to master file
    this calls get_lum from phot_modules.py
    I've edited this to account for fesc
    data_folder doesn't matter, I edited the relevant function...
    """

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num
            
        # file new values will be added to
        filename = "/cosma7/data/dp004/dc-seey1/data/flares/myversion/flares.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num = '00'

    else:
        ValueError(F"No input option of {inp}")

    print(f'filters: {filters}')

    lumfesc = get_lum(num, kappa, tag, BC_fac, filters = filters, LF = False,
                      inp = inp, Type = 'Total-random-fesc', log10t_BC = log10t_BC,
                      extinction = extinction, data_folder = data_folder,
                      aperture = aperture, fescs=fescs)

    print("lumfesc output array dimensions:",lumfesc.shape)
    print('lumfesc:', lumfesc)
    print('lumfesc[0]:', lumfesc[0])

    fl = flares.flares(fname = filename, sim_type = sim_type)

    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI_fesc")

    for ii, jj in enumerate(filters):

        _filter = jj[8:]

        fl.create_dataset(values = lumfesc[:,ii], name = F"{_filter}", #values = lumfesc[,ii]
                          group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI_fesc",
                          desc = F"Dust corrected luminosity (using ModelI) of the galaxy in the {_filter} band with a birth cloud factor of {BC_fac} following {extinction} curve and a random escape fraction",
                          unit = "ergs/s/Hz", overwrite=True)


def flux_write_out(num, tag, kappa, BC_fac, filters = flare.filters.ACS,
                   inp = 'FLARES', extinction = 'default', data_folder = 'data',
                   aperture='30'):

    """
    add flux to master file
    I haven't edited this, as I don't need the fluxes
    """

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

    fluxintr = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp,
                        Type = 'Intrinsic', extinction = extinction,
                        data_folder = data_folder, aperture = aperture)
    fluxstell = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp,
                         Type = 'Pure-stellar', extinction = extinction,
                         data_folder = data_folder, aperture = aperture)
    fluxBC = get_flux(num, 0, tag, BC_fac, filters = filters, inp = inp,
                      Type = 'Only-BC', extinction = extinction,
                      data_folder = data_folder, aperture = aperture)
    fluxatt = get_flux(num, kappa, tag, BC_fac, filters = filters, inp = inp,
                       Type = 'Total', extinction = extinction,
                       data_folder = data_folder, aperture = aperture)

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


def line_write_out(num, lines, tag, kappa, BC_fac, label, inp = 'FLARES',
                   LF = False, log10t_BC = 7., Type = 'Total', extinction = 'default',
                   data_folder = 'data', aperture='30', fescs=None):

    """
    add lines to mater file
    I've edited this to account for fesc
    """
    
    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = "/cosma7/data/dp004/dc-seey1/data/flares/myversion/flares.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num='00'

    else:
        ValueError(F"No input option of {inp}")

    calc = partial(get_lines, sim=num, kappa = kappa, tag = tag, BC_fac = BC_fac,
                   inp = inp, IMF = 'Chabrier_300', LF = False, log10t_BC = log10t_BC,
                   Type = Type, extinction = extinction, data_folder = data_folder,
                   aperture = aperture, fescs = fescs)

    pool = schwimmbad.MultiPool(processes=8)
    dat = np.array(list(pool.map(calc, lines)))
    pool.close()

    for ii, line in enumerate(lines):
        out_lum = dat[:,0][ii]
        out_EW = dat[:,1][ii]


        if Type == 'Total-random-fesc':
            fl = flares.flares(fname = filename, sim_type = sim_type)
            fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI_fesc", verbose=True)
            print (F'{line} is being written to disk')
            fl.create_dataset(values = out_lum, name = F"{line}/Luminosity",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI_fesc",
            desc = F"Dust corrected luminosity (using ModelI) of the galaxy with a birth cloud factor of {BC_fac} following {extinction} curve and random escape fraction.", unit = "ergs/s", overwrite=True)

            fl.create_dataset(values = out_EW, name = F"{line}/EW",
            group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Lines/DustModelI_fesc",
            desc = F"EW (using ModelI) of the galaxy with a birth cloud factor of {BC_fac} following {extinction} curve and random escape fraction", unit = "Angstrom", overwrite=True)



def sed_write_out(num, tag, kappa, BC_fac, inp = 'FLARES', IMF = 'Chabrier_300',
                  log10t_BC = 7., extinction = 'default', data_folder = 'data',
                  aperture='30', fescs=None):

    """
    add SEDs to master file
    I've edited this to account for fesc
    """

    if inp == 'FLARES':
        num = str(num)
        if len(num) == 1:
            num =  '0'+num

        filename = "/cosma7/data/dp004/dc-seey1/data/flares/myversion/flares.hdf5"
        sim_type = inp

    elif (inp == 'REF') or (inp == 'AGNdT9'):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
        num='00'

    else:
        ValueError(F"No input option of {inp}")

    try:
        dat = get_SED(num, kappa, tag, BC_fac, inp=inp, log10t_BC=log10t_BC,
                      extinction=extinction, data_folder=data_folder,
                      aperture=aperture, fescs=fescs)
        dust_fesc = dat[4]
    except:
        dust_fesc    = np.array([])

    fl = flares.flares(fname = filename, sim_type = sim_type)
    fl.create_group(F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED")

    #fl.create_dataset(values = lam, name = F"Wavelength",
    #group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    #desc = F"Wavelength array for the SED", unit = "Angstrom", overwrite=True)

    fl.create_dataset(values = dust_fesc, name = F"DustModelI_fesc",
    group = F"{tag}/Galaxy/BPASS_2.2.1/Chabrier300/SED",
    desc = F"SED from DustModelI with kappa_BC={BC_fac} and {extinction} curve and random escape fraction", unit = "ergs/s/Hz", overwrite=True)




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

    #Parameters used
    BC_fac = 1.
    kappa = 0.0795
    aperture = '30'
    
    sim_type = get_simtype(inp)
    fl = flares.flares(fname = 'tmp', sim_type = sim_type)
    tags = fl.tags
    print (num, tag, inp, data_folder)

    with h5py.File('/cosma7/data/dp004/dc-seey1/data/flares/myversion/flares.hdf5', 'r') as myfile:
        fescs = myfile[f'{num}/{tag}/Galaxy/fesc_random']

    if prop == 'Luminosity':
        print ("Calculating luminosities")
        lum_write_out(num=num, tag = tag, kappa = kappa, BC_fac = BC_fac, inp = inp, data_folder = data_folder, filters=flare.filters.TH[:-1], aperture=aperture, fescs=fescs)

    elif prop == 'Flux':
        print ("Calculating fluxes")
        flux_write_out(num=num, tag = tag, kappa = kappa, BC_fac = BC_fac,
                       filters = flare.filters.ACS + flare.filters.WFC3NIR_W + flare.filters.IRAC + flare.filters.Euclid + flare.filters.Subaru + flare.filters.NIRCam + flare.filters.MIRI,
                       inp = inp, data_folder = data_folder, aperture=aperture)

    elif prop == 'Lines':
        print ("Calculating line luminosities and EW")
        m=models.EmissionLines("BPASSv2.2.1.binary/Chabrier_300", verbose=False)
        lines = m.lines

        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Total', inp = inp, LF = False, log10t_BC = 7., label = 'Total', data_folder = data_folder, aperture=aperture)
        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Intrinsic', inp = inp, LF = False, log10t_BC = 7., label = 'Intrinsic', data_folder = data_folder, aperture=aperture)
        line_write_out(num, lines, tag = tag, kappa = kappa, BC_fac = BC_fac, Type = 'Only-BC', inp = inp, LF = False, log10t_BC = 7., label = 'No_ISM', data_folder = data_folder, aperture=aperture)

    elif prop == 'SED':
        print ("Calculating SEDs")
        sed_write_out(num, tag, kappa, BC_fac, inp = inp, IMF = 'Chabrier_300', log10t_BC = 7., data_folder = data_folder, aperture=aperture)

    else:
        ValueError(F"No input of type {prop}")
