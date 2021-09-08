import numpy as np
import h5py
import sys

ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

num = str(ii)
tag = str(tag)
inp = str(inp)
data_folder = str(data_folder)

num = "%02d"%int(num)



if inp == 'FLARES':
    if len(num) == 1:
        num =  '0'+num
    filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
    sim_type = 'FLARES'


elif inp == 'REF' or inp == 'AGNdT9':
    filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
    sim_type = 'PERIODIC'

else:
    ValueError("Type of input simulation not recognized")


with h5py.File(filename, 'r') as hf:
    dindex = np.array(hf[tag+'/Particle'].get('DM_Index'), dtype = np.int64)
    sindex = np.array(hf[tag+'/Particle'].get('S_Index'), dtype = np.int64)
    gindex = np.array(hf[tag+'/Particle'].get('G_Index'), dtype = np.int64)
    dnum =   np.array(hf[tag+'/Galaxy'].get('DM_Length'), dtype = np.int64)
    snum =   np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
    gnum =   np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)


from download_methods import recalculate_derived_subhalo_properties, save_to_hdf5, get_recent_SFR


SMass, GMass, DMass, total_SFR = \
    recalculate_derived_subhalo_properties(inp, num, tag, snum, gnum, dnum, sindex, gindex, dindex)


save_to_hdf5(num, tag, SMass, 'Mstar', 'Total stellar mass of the subhalo', group='Galaxy', inp=inp)
save_to_hdf5(num, tag, GMass, 'Mgas', 'Total gas mass of the subhalo', group='Galaxy', inp=inp)
save_to_hdf5(num, tag, DMass, 'Mdm', 'Total dark matter mass of the subhalo', group='Galaxy', inp=inp)
save_to_hdf5(num, tag, total_SFR, 'SFR', 'Total instantaneous star formation rate of the subhalo', group='Galaxy', inp=inp)


timescales = [1,5,10,20,40,100]
aperture_sizes = [1, 3, 5, 10, 20, 30, 40, 50, 70 , 100]
SFR, inst_SFR, Mstar = get_recent_SFR(num, tag, t = timescales, aperture_size = aperture_sizes, inp = inp)

for _ap in aperture_sizes:
    save_to_hdf5(num, tag, Mstar[_ap], f'Mstar_{_ap}', 
                 f'Stellar mass contained within a {_ap} Mpc aperture', 
                 group=f'Galaxy/Mstar_aperture', inp=inp)
    
    save_to_hdf5(num, tag, inst_SFR[_ap], f'SFR_inst', 
                 f'Instantaneous star formation rate contained within a {_ap} Mpc aperture', 
                 group=f'Galaxy/SFR_aperture/SFR_{_ap}', inp=inp)

    for _t in timescales:
        save_to_hdf5(num, tag, SFR[_ap][_t], f'SFR_{_t}_Myr', 
                     f'Star formation rate measured over the past {_t} Myr in a {_ap} Mpc aperture', 
             group=f'Galaxy/SFR_aperture/SFR_{_ap}', inp=inp)
        



