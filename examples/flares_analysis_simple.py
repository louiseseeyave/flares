
import numpy as np
import matplotlib.cm as cm

import flares
import flares_analysis
import flare.plt as fplt

# ----------------------------------------------------------------------
# --- open data

fl = flares.flares('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5', sim_type='FLARES')
halo = fl.halos

# ----------------------------------------------------------------------
# --- define parameters and tag
tag = fl.tags[-3]  # --- select tag -3 = z=7
log10Mstar_limit = 9.


# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': 'SFR_inst_30', 'name': None, 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': 'Mstar_30', 'name': None, 'log10': True})

# --- get quantities (and weights and deltas)
D = flares_analysis.get_datasets(fl, tag, quantities)

# ----------------------------------------------
# define new quantities
D['log10sSFR'] = np.log10(D['SFR_inst_30'])-np.log10(D['Mstar_30'])+9

# ----------------------------------------------
# define selection
s = D['log10Mstar_30']>log10Mstar_limit

# ----------------------------------------------
# Print number of galaxies meeting the selection
print(f"Total number of galaxies: {len(D['log10Mstar_30'][s])}")


# ----------------------------------------------
# ----------------------------------------------
# plot with colour bar


# --- get default limits and modify them to match the selection range
limits = flares_analysis.limits
limits['log10Mstar_30'] = [log10Mstar_limit, 10.9]

# --- get default limits and modify them to match the selection range
labels = flares_analysis.labels
labels['log10sSFR'] = 'specific\ star\ formation\ rate/Gyr^{-1}' # this is just to demonstrate how to use this


x = 'log10Mstar_30'
y = 'log10sSFR'

# --- make plot with colour bar plot
fig = flares_analysis.simple(D, x, y, s, limits = limits, labels = labels)


fig.savefig(f'simple.pdf')
