import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
from flares_utility import analyse

# test whether the particle lengths make sense


master = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'

a = analyse.analyse(master)

regions = np.array(['00','01','02','03','04','05','06','07','08','09','10',
                    '11','12','13','14','15','16','17','18','19','20','21',
                    '22','23','24','25','26','27','28','29','30','31','32',
                    '33','34','35','36','37','38','39'])
# snapshots = np.array(['000_z015p000','001_z014p000','002_z013p000','003_z012p000',
#                       '004_z011p000','005_z010p000','006_z009p000','007_z008p000',
#                       '008_z007p000','009_z006p000','010_z005p000','011_z004p770'])

snapshots = np.array(['011_z004p770'])


# --------------------------------------------------------------------

# plot stellar particle count while performing the check

rows = 3
cols = 3
s_fig, s_axs = plt.subplots(rows, cols, sharex=True, sharey=True,
                            figsize=(3*cols,3*rows))

for snap in snapshots:
    
    slen_subfind = []
    slen_notsubfind = []
    glen_subfind = []
    glen_notsubfind = []
    gal_ids = []

    for rg in regions:

        with h5py.File(master, 'r') as m:
            s_ap = a.load_aperture_mask(rg, snap, particle_type='Star',
                                        aperture='30', return_dict=True)
            g_ap = a.load_aperture_mask(rg, snap, particle_type='Gas',
                                        aperture='30', return_dict=True)
            mstar = a.load_single_dataset(rg, snap,'Galaxy/Mstar_aperture','30')
            mstar = np.log10(mstar) + 10
            keep = mstar > 8
            skeys = np.array(list(s_ap.keys()))[keep]
            gkeys = np.array(list(g_ap.keys()))[keep]
            if np.any(skeys != gkeys):
                print(f'something is wrong...  skeys != gkeys')
            slen = [np.sum(s_ap[i]) for i in skeys]
            glen = [np.sum(g_ap[i]) for i in gkeys]
            slen_subfind.extend(slen)
            glen_subfind.extend(glen)

            new_gal_ids = [rg+'_'+snap+'_'+str(i) for i in skeys]
            gal_ids.extend(new_gal_ids)
            

        with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc.pkl', 'rb') as f:
            partdata = pickle.load(f)
            sn = partdata['stellar_neighbours']
            gn = partdata['gas_neighbours']
            sn_keys = np.array(list(sn.keys()))[keep]
            gn_keys = np.array(list(gn.keys()))[keep]
            if np.any(sn_keys != gn_keys):
                print(f'something is wrong...  sn_keys != gn_keys')
            slen = [len(sn[i]) for i in sn_keys]
            glen = [len(gn[i]) for i in gn_keys]
            slen_notsubfind.extend(slen)
            glen_notsubfind.extend(glen)

    print(f'for snaphot {snap}:')

    # get row and column number
    z = float(snap[5:].replace('p','.'))
    z_temp = np.floor(z)
    r = int((12-z_temp)//3)
    c = int(12 - z_temp - 3*r)
    
    s_check = np.array(slen_notsubfind) >= np.array(slen_subfind)
    if np.sum(~s_check)==0:
        print('all good for the stellar particles!')
    else:
        print(f'hmm. there are {np.sum(~s_check)} out of {len(s_check)} galaxies with weird s lengths')
        s_ind = np.where(s_check==False)[0]
        print('first few ids of weird cases:', np.array(gal_ids)[s_ind[0:5]])
        s_diff = np.array(slen_subfind)[s_ind] - np.array(slen_notsubfind)[s_ind]
        s_max = np.argmax(s_diff)
        gal_id = (np.array(gal_ids)[s_ind])[s_max]
        print('check out this galaxy...', gal_id)
        print('the difference in s count is', s_diff[s_max])

    g_check = np.array(glen_notsubfind) >= np.array(glen_subfind)
    if np.sum(~g_check)==0:
        print('all good for the gas particles!')
    else:
        print(f'hmm. there are {np.sum(~g_check)} out of {len(s_check)} galaxies with weird g lengths')
        g_ind = np.where(g_check==False)[0]
        print('first few ids of weird cases:', np.array(gal_ids)[g_ind[0:5]])
        g_diff = np.array(glen_subfind)[g_ind] - np.array(glen_notsubfind)[g_ind]
        g_max = np.argmax(g_diff)
        gal_id = (np.array(gal_ids)[g_ind])[g_max]
        print('check out this galaxy...', gal_id)
        print('the difference in g count is', g_diff[g_max])
        
    if z < 13:
        s_axs[r,c].scatter(slen_notsubfind, slen_subfind, marker='.', alpha=0.4,
                           edgecolors='none')
        s_axs[r,c].plot([0,100000],[0,100000], color='grey', ls='--', lw=.5)
        s_axs[r,c].text(.96,.9,f'z={z}',ha="right",va="top",transform=s_axs[r,c].transAxes)
        s_axs[r,c].tick_params(direction='in')
        
    # plt.scatter(slen_notsubfind, slen_subfind, marker='.', alpha=0.4,
    #             edgecolors='none')
    # plt.xlabel('n stellar particles (30kpc)')
    # plt.ylabel('n stellar particles (subfind)')
    # plt.plot([np.amin(slen_subfind), np.amax(slen_subfind)],
    #          [np.amin(slen_subfind), np.amax(slen_subfind)],
    #          color='grey')
    # plt.savefig(f'figures/{snap}_slen', dpi=300)
    
    # plt.close()

    # plt.scatter(glen_notsubfind, glen_subfind, marker='.', alpha=0.4,
    #             edgecolors='none')
    # plt.xlabel('n gas particles (30kpc)')
    # plt.ylabel('n gas particles (subfind 30kpc)')
    # plt.plot([np.amin(glen_subfind), np.amax(glen_subfind)],
    #          [np.amin(glen_subfind), np.amax(glen_subfind)],
    #          color='grey')
    # plt.savefig(f'figures/{snap}_glen', dpi=300)

    # plt.close()


s_fig.subplots_adjust(wspace=0, hspace=0)

# add a big axis, hide frame
s_fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tight_layout(pad=0.4, w_pad=5, h_pad=0.5)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('n stellar particles (30kpc)')
plt.ylabel('n stellar particles (subfind 30kpc)')
# plt.savefig('figures/slen_z4-12', dpi=300)

plt.close()

# --------------------------------------------------------------------

# plot gas particle count (no checks)

rows = 3
cols = 3
g_fig, g_axs = plt.subplots(rows, cols, sharex=True, sharey=True,
                            figsize=(3*cols,3*rows))


for snap in snapshots:
    
    glen_subfind = []
    glen_notsubfind = []

    for rg in regions:

        with h5py.File(master, 'r') as m:
            g_ap = a.load_aperture_mask(rg, snap, particle_type='Gas',
                                        aperture='30', return_dict=True)
            mstar = a.load_single_dataset(rg, snap,'Galaxy/Mstar_aperture','30')
            mstar = np.log10(mstar) + 10
            keep = mstar > 8
            gkeys = np.array(list(g_ap.keys()))[keep]
            glen = [np.sum(g_ap[i]) for i in gkeys]
            glen_subfind.extend(glen)

        with open(f'data_valid_particles/flares_{rg}_{snap}_30pkpc.pkl', 'rb') as f:
            partdata = pickle.load(f)
            gn = partdata['gas_neighbours']
            gn_keys = np.array(list(gn.keys()))[keep]
            glen = [len(gn[i]) for i in gn_keys]
            glen_notsubfind.extend(glen)

    # get row and column number
    z = float(snap[5:].replace('p','.'))
    z_temp = np.floor(z)
    r = int((12-z_temp)//3)
    c = int(12 - z_temp - 3*r)

    if z < 13:
        print(f'plotting snaphot {snap}')
        g_axs[r,c].scatter(glen_notsubfind, glen_subfind, marker='.', alpha=0.4,
                           edgecolors='none')
        g_axs[r,c].plot([0,100000],[0,100000], color='grey', ls='--', lw=.5)
        g_axs[r,c].text(.96,.9,f'z={z}',ha="right",va="top",transform=g_axs[r,c].transAxes)
        g_axs[r,c].tick_params(direction='in')

g_fig.subplots_adjust(wspace=0, hspace=0)

# add a big axis, hide frame
g_fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tight_layout(pad=0.4, w_pad=5, h_pad=0.5)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('n gas particles (30kpc)')
plt.ylabel('n gas particles (subfind 30kpc)')
# plt.savefig('figures/glen_z4-12', dpi=300)
