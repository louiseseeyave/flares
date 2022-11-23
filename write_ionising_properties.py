import numpy as np
import h5py
import sys
from synthobs.sed import models
from download_methods import save_to_hdf5



def calculate_U(Q_avg, n_h=100):

    """
    Get ionisation parameter
    (Taken from synthesizer, can just import the function in the future)
    
    Args
    Q - units: s^-1
    n_h - units: cm^-3
    Returns
    U - units: dimensionless
    """
    
    alpha_B = 2.59e-13 # cm^3 s^-1
    c_cm = 2.99e8 * 100 # cm s^-1
    epsilon = 1.

    return ((alpha_B**(2./3)) / c_cm) *\
            ((3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi))**(1./3)

    
if __name__ == "__main__":

    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    num = str(ii)
    tag = str(tag)
    inp = str(inp)
    data_folder = str(data_folder)

    print(f'region {ii}, snapshot {tag}')


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

        sindex = np.array(hf[tag+'/Particle'].get('S_Index'), dtype = np.int64)
        snum = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        S_mass = np.array(hf[F'{tag}/Particle'].get('S_MassInitial'),
                          dtype = np.float64)*1e10 # Msun
        S_age = np.array(hf[F'{tag}/Particle'].get('S_Age'),
                         dtype = np.float64)*1e3 # Gyr -> Myr
        S_Z = np.array(hf[F'{tag}/Particle'].get('S_Z_smooth'),
                       dtype = np.float64)
        S_ap_bool = np.array(hf[F'{tag}/Particle/Apertures/Star'].get('30'),
                             dtype = np.float64) # 30pkpc aperture
        L_uv = np.array(hf[F'{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Pure_Stellar'].get('FUV'),
                        dype = np.float64) # ergs/s/Hz
        L_uv = np.log10(L_uv)

    
    sbegin = np.zeros(len(snum), dtype = np.int64)
    send = np.zeros(len(snum), dtype = np.int64)
    sbegin[1:] = np.cumsum(snum)[:-1]
    send = np.cumsum(snum)

    log10Q = np.zeros(len(sbegin))
    log10xi = np.zeros(len(sbegin))
    U = np.zeros(len(sbegin))

    model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300')
    
    for jj in range(len(sbegin)):

        this_age = S_age[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_smass = S_mass[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]
        this_sZ = S_Z[sbegin[jj]:send[jj]][S_ap_bool[sbegin[jj]:send[jj]]]

        # get ionising emissivity [log10(s^-1)]
        log10Q[jj] = models.generate_log10Q(model, this_smass, this_age, this_sZ)
        # get ionising photon production efficiency [log10(erg^-1 Hz)]
        log10xi[jj] = log10Q[jj] - L_uv[jj]
        # get ionisation parameter
        U[jj] = calculate_U(10**log10Q[jj])

    save_to_hdf5(num, tag, log10Q, 'IonisingEmissivity', 'Ionising emissivity',
                 group=f'Galaxy', inp=inp, unit='log10(s^-1)', data_folder=data_folder, overwrite=True)

    save_to_hdf5(num, tag, log10xi, 'IonisingPPE', 'Ionising photon production efficiency',
                 group=f'Galaxy', inp=inp, unit='log10(erg^-1 Hz)', data_folder=data_folder, overwrite=True)

    save_to_hdf5(num, tag, U, 'IonisationParameter', 'Ionising parameter',
                 group=f'Galaxy', inp=inp, unit='None', data_folder=data_folder, overwrite=True)
