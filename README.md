# First Light And Reionisation Epoch Simulations (FLARES)

A python convenience module for working with FLARES data.

## Requirements

- numpy
- h5py
- scipy 
- [numba](https://numba.readthedocs.io/en/stable/user/installing.html)
- schwimmbad
- [eagle_IO](https://github.com/flaresimulations/eagle_IO)
- [SynthObs](https://github.com/stephenmwilkins/SynthObs) 

## Installation

Run the following to add this module *permanently* to your path:

    export PYTHONPATH=$PYTHONPATH:/path/to/this/directory/flares

You can then just run

    import flares

in any other scripts to get the flares class and associated functionality.

## Set up and data location

The original FLARES data on COSMA are located here:

    /cosma7/data/dp004/FLARES/FLARES-1

You may need to update this location in `flares.py#L29` by changing the `self.directory` string.


## Tutorial

`flares.py` contains the `flares` class, which contains a lot of useful functionality for analysing the resims, such as the specified halos (`flares.halos`) and snapshots (`flares.tags`) you wish to analyse; these should be updated as new resims are completed.

In order to make working with the data easier we typically download a subset of subhalo and particle arrays to a 'master' HDF5 file. 
This can then be used in place of the raw simulation outputs.
It also typically includes other derived properties, such as stellar masses and SFRs within apertures, and forward modelled predictions for emission.
Most user will typically use a master file that has already been created (please speak to one of the team to get access), however for reference below we specify how to create a master file from scratch.
Each bash submission script may require modifying for your particular architecture and python environment.

- First we download the particle properties, handled in `run_download_particles.cosma.sh`
- We can then download subhalo properties, handled in `run_download_properties.cosma.sh`.  The required arrays can be defined in `req_arrays.txt`
- If you wish to calculate emission from subhalos, you first need to calculate the line of sigh metal column density, handled in `run_calc_Zlos.cosma.sh`
- You can then run `run_photometry.cosma.sh` to generate predicted emission. Within this script you can select whether you require fluxes, luminosities, full SEDs or just line information.
- Finally, the outputs from each region can be combined using `combine_master_file.py` 

Once this has completed you will have a single file `data/flares.hdf5` with the following (rough) data structure: `Resim_num/Property_type/Property`, where `Resim_num` is the number of resims (see [here](https://docs.google.com/spreadsheets/d/1NzQee05rNCml1YEKXuD8L9JOW5Noh8oj9K9bcS2RQlY/edit?usp=sharing)), `Property_type` can be either Galaxy (like stellar mass, sfr, etc) or Particle (individual properties of gas/stellar particles) and `Property` is the required property. 

### Example

Once the data is downloaded, you can use it as so,

```
import flares
fl = flares.flares('./data/flares.hdf5', sim_type='FLARES')

mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy')

halo = fl.halos[0]
tag = fl.tags[0]

print (mstar[halo][tag][:10])
```

Creating distribution functions, e.g: stellar mass function for z=5:

```
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import flares

fl = flares.flares('./data/flares.hdf5', sim_type='FLARES')
halo = fl.halos
tag = fl.tags[-1]
volume = (4/3)*np.pi*(fl.radius**3)

mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy')
df = pd.read_csv('weight_files/weights_grid.txt')
weights = np.array(df['weights'])

bins = np.arange(8, 11.5, 0.2)
bincen = (bins[1:]+bins[:-1])/2.
binwidth = bins[1:] - bins[:-1]

hist = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)

for ii in range(len(weights)):
    tmp, bin_edges = np.histogram(np.log10(mstar[halo[ii]][tag]), bins = bins)
    hist+=tmp*weights[ii]
    err+=np.square(np.sqrt(tmp)*weights[ii])
    
smf = hist/(volume*binwidth)
smf_err = np.sqrt(err)/(volume*binwidth)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), facecolor='w', edgecolor='k')

y_lo, y_up = np.log10(smf)-np.log10(smf-smf_err), np.log10(smf+smf_err)-np.log10(smf)
axs.errorbar(bincen, np.log10(smf), yerr=[y_lo, y_up], ls='', marker='o', label=rF"Flares $z={float(tag[5:].replace('p','.'))}$")

axs.set_ylabel(r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$', fontsize=14)
axs.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\star}/\mathrm{M_{\odot}})$', fontsize=14)
axs.set_xlim((8, 11.4))
axs.set_ylim((-8.2, -0.8))
axs.set_xticks(np.arange(8., 11.5, 1))
axs.grid(True, alpha = 0.5)
axs.legend(frameon=False, fontsize = 14, numpoints=1, ncol = 2)
axs.minorticks_on()
axs.tick_params(axis='x', which='minor', direction='in')
axs.tick_params(axis='y', which='minor', direction='in')
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(13)

plt.show()
```

Extracting stellar particle information,

```
import numpy as np
import h5py
fname = './data/flares.hdf5'
num = '00'
with h5py.File(fname, 'r') as hf:
    S_len = np.array(hf[num+'/'+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
    S_mass = np.array(hf[num+'/'+tag+'/Particle'].get('S_Mass'), dtype = np.float64)
    S_Z = np.array(hf[num+'/'+tag+'/Particle'].get('S_Z'), dtype = np.float64)
    S_age = np.array(hf[num+'/'+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3

begin = np.zeros(len(S_len), dtype = np.int64)
end = np.zeros(len(S_len), dtype = np.int64)
begin[1:] = np.cumsum(S_len)[:-1]
end = np.cumsum(S_len)

#Age of all particles belonging to first galaxy in resim region 'num'

print (S_age[begin[0]:end[0]])
```
