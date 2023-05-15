import numpy as np
import h5py
import sys

def DTM_fit(Z, Age):

    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5/(D0*Z)
    DTM = D0 + (D1-D0)*(1.-np.exp(-alpha*(Z**beta)*((Age/(1e3*tau))**gamma)))
    if ~np.isfinite(DTM): DTM = 0.

    return DTM

if __name__ == "__main__":

    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    num = str(ii)
    tag = str(tag)
    inp = str(inp)
    data_folder = str(data_folder)


    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        sim_type = 'FLARES'


    elif (inp == 'REF') or (inp == 'AGNdT9') or ('RECAL' in inp):
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'

    else:
        ValueError("Type of input simulation not recognized")


    with h5py.File(filename, 'r') as hf:
        dindex  = np.array(hf[tag+'/Particle'].get('DM_Index'), dtype = np.int64)
        sindex  = np.array(hf[tag+'/Particle'].get('S_Index'), dtype = np.int64)
        gindex  = np.array(hf[tag+'/Particle'].get('G_Index'), dtype = np.int64)
        bhindex = np.array(hf[tag+'/Particle'].get('BH_Index'), dtype = np.int64)
        dnum    = np.array(hf[tag+'/Galaxy'].get('DM_Length'), dtype = np.int64)
        snum    = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        gnum    = np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        bhnum   = np.array(hf[tag+'/Galaxy'].get('BH_Length'), dtype = np.int64)


    from download_methods import recalculate_derived_subhalo_properties, save_to_hdf5, get_recent_SFR, get_aperture_inst_SFR


    # SMass, GMass, DMass, total_SFR = \

    SMass, GMass, BHMass, DMass = \
       recalculate_derived_subhalo_properties(inp, num, tag, snum, gnum, dnum, bhnum, sindex, gindex, bhindex, dindex, data_folder=data_folder)

    # uncomment later ------------------------------------------------
    # save_to_hdf5(num, tag, SMass, 'Mstar', 'Total stellar mass of the subhalo', group='Galaxy', inp=inp,
    #                 data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, GMass, 'Mgas', 'Total gas mass of the subhalo', group='Galaxy', inp=inp,
    #                 data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, BHMass, 'Mbh', 'Total BH mass of the subhalo', group='Galaxy', inp=inp,
    #                 data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, DMass, 'Mdm', 'Total dark matter mass of the subhalo', group='Galaxy', inp=inp,
    #                data_folder=data_folder, overwrite=True)
    # ----------------------------------------------------------------
    # save_to_hdf5(num, tag, total_SFR, 'SFR',
    #              'Total instantaneous star formation rate of the subhalo', group='Galaxy', inp=inp)


    timescales = [1,5,10,20,50,100,200]
    aperture_sizes = [1, 3, 5, 10, 20, 30, 40, 50, 70, 100, 1e4]
    aperture_labels = np.hstack([aperture_sizes[:-1], 'total'])
    SFR, Mstar, Mgas, S_ap_bool, G_ap_bool, BH_ap_bool = get_recent_SFR(num,tag,t=timescales,aperture_size=aperture_sizes,inp=inp, data_folder=data_folder)
    inst_SFR = get_aperture_inst_SFR(num,tag,aperture_size=aperture_sizes,inp=inp, data_folder=data_folder)

    # uncomment later ------------------------------------------------
    # for jj,_ap in enumerate(aperture_sizes[:-1]):
    #     save_to_hdf5(num, tag, Mstar[_ap], f'{_ap}',
    #                  f'Stellar mass contained within a {_ap} pkpc aperture',
    #                  group=f'Galaxy/Mstar_aperture', inp=inp, unit='1E10 Msun', data_folder=data_folder, overwrite=True)

    #     save_to_hdf5(num, tag, Mgas[_ap], f'{_ap}',
    #                  f'Gas mass contained within a {_ap} pkpc aperture',
    #                  group=f'Galaxy/Mgas_aperture', inp=inp, unit='1E10 Msun', data_folder=data_folder, overwrite=True)

    #     save_to_hdf5(num, tag, inst_SFR[_ap], f'inst',
    #                  f'Instantaneous star formation rate contained within a {_ap} pkpc aperture',
    #                  group=f'Galaxy/SFR_aperture/{_ap}', inp=inp, unit='Msun/yr', data_folder=data_folder, overwrite=True)

    #     ## save aperture boolean selection
    #     save_to_hdf5(num, tag, S_ap_bool[jj], F'{_ap}',
    #               f'Boolean array of star particles within {_ap} pkpc aperture', group='Particle/Apertures/Star', inp=inp, unit='bool', data_folder=data_folder, overwrite=True)

    #     save_to_hdf5(num, tag, G_ap_bool[jj], F'{_ap}',
    #                f'Boolean array of gas particles within {_ap} pkpc aperture', group='Particle/Apertures/Gas', inp=inp, unit='bool', data_folder=data_folder, overwrite=True)

    #     save_to_hdf5(num, tag, BH_ap_bool[jj], F'{_ap}',
    #                f'Boolean array of BH particles within {_ap} pkpc aperture', group='Particle/Apertures/BH', inp=inp, unit='bool', data_folder=data_folder, overwrite=True)

    #     for _t in timescales:
    #         save_to_hdf5(num, tag, SFR[_ap][_t], f'{_t}Myr',
    #                      f'Star formation rate measured over the past {_t} Myr in a {_ap} pkpc aperture',
    #              group=f'Galaxy/SFR_aperture/{_ap}', inp=inp, unit='Msun/yr', data_folder=data_folder, overwrite=True)


    # ## save total SFR
    # _ap = 1e4
    # save_to_hdf5(num, tag, inst_SFR[_ap], f'inst',
    #              f'Total instantaneous star formation rate in the subhalo',
    #              group='Galaxy/SFR_total', inp=inp, unit='Msun/yr', data_folder=data_folder, overwrite=True)

    # for _t in timescales:
    #     save_to_hdf5(num, tag, SFR[_ap][_t], f'{_t}Myr',
    #                  f'Total star formation rate measured over the past {_t} Myr in the subhalo',
    #                  group='Galaxy/SFR_total', inp=inp, unit='Msun/yr', data_folder=data_folder, overwrite=True)
    # ----------------------------------------------------------------



    ## Extra averaged information wihtin 30 pkpc
    S_ap_bool = S_ap_bool[5]
    G_ap_bool = G_ap_bool[5]

    with h5py.File(filename, 'r') as hf:
        S_mass          = np.array(hf[F'{tag}/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        S_currentmass   = np.array(hf[F'{tag}/Particle'].get('S_Mass'), dtype = np.float64)*1e10
        G_mass          = np.array(hf[F'{tag}/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        S_age           = np.array(hf[F'{tag}/Particle'].get('S_Age'), dtype = np.float64) #Age is in Gyr,
        S_Z             = np.array(hf[F'{tag}/Particle'].get('S_Z_smooth'), dtype = np.float64)
        G_Z             = np.array(hf[F'{tag}/Particle'].get('G_Z_smooth'), dtype = np.float64)
        O_abundance     = np.array(hf[F'{tag}/Particle'].get('S_O_Abundance_smooth'), dtype = np.float64)
        Fe_abundance    = np.array(hf[F'{tag}/Particle'].get('S_Fe_Abundance_smooth'), dtype = np.float64)
        G_SFR           = np.array(hf[F'{tag}/Particle'].get('G_SFR'), dtype = np.float64)


    sbegin = np.zeros(len(snum), dtype = np.int64)
    send = np.zeros(len(snum), dtype = np.int64)
    sbegin[1:] = np.cumsum(snum)[:-1]
    send = np.cumsum(snum)

    gbegin = np.zeros(len(gnum), dtype = np.int64)
    gend = np.zeros(len(gnum), dtype = np.int64)
    gbegin[1:] = np.cumsum(gnum)[:-1]
    gend = np.cumsum(gnum)

    S_massweightedage   = np.zeros(len(sbegin))
    S_massweightedZ     = np.zeros(len(sbegin))
    YS_massweightedZ    = np.zeros(len(sbegin))

    S_currentmassweightedage   = np.zeros(len(sbegin))
    S_currentmassweightedZ     = np.zeros(len(sbegin))

    G_massweightedZ     = np.zeros(len(sbegin))
    DTM                 = np.zeros(len(sbegin))

    S_massweightedO            = np.zeros(len(sbegin))
    S_currentmassweightedO     = np.zeros(len(sbegin))
    S_massweightedFe           = np.zeros(len(sbegin))
    S_currentmassweightedFe    = np.zeros(len(sbegin))

    G_SFRweightedZ  = np.zeros(len(sbegin))

    for jj in range(len(gbegin)):

        this_age    = S_age[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        young_stars = this_age < 10 # keep young stars (<10Myr)
        this_smass  = S_mass[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_ysmass = this_smass[young_stars]
        this_scmass = S_currentmass[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_gmass  = G_mass[gbegin[jj]:gend[jj]][G_ap_bool[gbegin[jj]:gend[jj]]]
        this_sZ     = S_Z[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_ysZ    = this_sZ[young_stars]
        this_gZ     = G_Z[gbegin[jj]:gend[jj]][G_ap_bool[gbegin[jj]:gend[jj]]]
        this_O      = O_abundance[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_Fe     = Fe_abundance[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_gsfr   = G_SFR[gbegin[jj]:gend[jj]][G_ap_bool[gbegin[jj]:gend[jj]]]

        S_massweightedage[jj]           = np.nansum(this_smass*this_age)/np.nansum(this_smass)
        S_massweightedZ[jj]             = np.sum(this_smass*this_sZ)/np.sum(this_smass)
        YS_massweightedZ[jj]            = np.sum(this_ysmass*this_ysZ)/np.sum(this_ysmass)
        S_currentmassweightedage[jj]    = np.nansum(this_scmass*this_age)/np.nansum(this_scmass)
        S_currentmassweightedZ[jj]      = np.sum(this_scmass*this_sZ)/np.sum(this_scmass)
        G_massweightedZ[jj]             = np.sum(this_gmass*this_gZ)/np.sum(this_gmass)

        S_massweightedO[jj]             = np.nansum(this_smass*this_O)/np.nansum(this_smass)
        S_currentmassweightedO[jj]      = np.nansum(this_scmass*this_O)/np.nansum(this_scmass)
        S_massweightedFe[jj]            = np.nansum(this_smass*this_Fe)/np.nansum(this_smass)
        S_currentmassweightedFe[jj]     = np.nansum(this_scmass*this_Fe)/np.nansum(this_scmass)
        G_SFRweightedZ[jj]              = np.nansum(this_gmass*this_gsfr)/np.nansum(this_gsfr)

        DTM[jj] = DTM_fit(np.nanmean(this_gZ), S_massweightedage[jj]*1e3)

    # save_to_hdf5(num, tag, S_massweightedage, 'MassWeightedStellarAge',
    #              'Initial mass-weighted stellar age within 30 pkpc aperture',
    #              group=f'Galaxy/StellarAges', inp=inp, unit='Gyr', data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, S_currentmassweightedage, 'CurrentMassWeightedStellarAge',
    #              'Current mass-weighted stellar age within 30 pkpc aperture',
    #              group=f'Galaxy/StellarAges', inp=inp, unit='Gyr', data_folder=data_folder, overwrite=True)

    # save_to_hdf5(num, tag, S_massweightedZ, 'MassWeightedStellarZ',
    #              'Initial mass-weighted stellar metallicity within 30 pkpc aperture',
    #              group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, S_currentmassweightedZ, 'CurrentMassWeightedStellarZ',
    #              'Current mass-weighted stellar metallicity within 30 pkpc aperture',
    #              group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    save_to_hdf5(num, tag, YS_massweightedZ, 'MassWeightedYoungStellarZ',
                 'Initial mass-weighted stellar metallicity of young stellar particles (<10Myr) within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    # save_to_hdf5(num, tag, G_massweightedZ, 'MassWeightedGasZ',
    #              'Mass-weighted gas-phase metallicity within 30 pkpc aperture',
    #              group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    save_to_hdf5(num, tag, G_SFRweightedZ, 'SFRWeightedGasZ',
                 'SFR-weighted gas-phase metallicity within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)

    # save_to_hdf5(num, tag, DTM, 'DTM',
    #              'Dust-to-metal ratio of the galaxy based on equation 15 in Vijayan et al. (2019)  within 30 pkpc aperture',
    #              group=f'Galaxy', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)

    save_to_hdf5(num, tag, S_massweightedO, 'MassWeightedStellarO',
                 'Initial mass-weighted stellar oxygen abundance within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    save_to_hdf5(num, tag, S_currentmassweightedO, 'CurrentMassWeightedStellarO',
                 'Current mass-weighted stellar oxygen abundance within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)

    save_to_hdf5(num, tag, S_massweightedFe, 'MassWeightedStellarFe',
                 'Initial mass-weighted stellar iron abundance within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
    save_to_hdf5(num, tag, S_currentmassweightedFe, 'CurrentMassWeightedStellarFe',
                 'Current mass-weighted stellar iron abundance within 30 pkpc aperture',
                 group=f'Galaxy/Metallicity', inp=inp, unit='No unit', data_folder=data_folder, overwrite=True)
