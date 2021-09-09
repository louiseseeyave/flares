import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import cmasher as cmr

import astropy.constants as constants
import astropy.units as units

import flares
import h5py
import flare.plt as fplt



fancy = lambda x: r'$\rm '+x.replace(' ','\ ')+'$'
ml = lambda x: r'$\rm '+x+'$'


deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])

log10x = lambda x: rf'\log_{{10}}({x})'

# default plot limits

labels = {}


labels['log10BB'] = r'\log_{10}(L_{4200}/L_{3500})'
labels['log10BB_Intrinsic'] = rf"{labels['log10BB']}^{{\rm int}}"
labels['log10BB_Pure_Stellar'] = rf"{labels['log10BB']}^{{\star}}"

labels['beta'] = r'\beta'
labels['beta_DustModelI'] = labels['beta']
labels['beta_Intrinsic'] = rf"{labels['beta']}^{{\rm int}}"
labels['beta_Pure_Stellar'] = rf"{labels['beta']}^{{\star}}"

labels['log10HbetaEW'] = r'\log_{10}(H\beta\ EW/\AA)'
labels['log10HbetaEW_Intrinsic'] = rf"{labels['log10HbetaEW']}^{{\rm int}}"




labels['log10SFR_inst_30'] = r'\log_{10}({\rm SFR_{inst}}/{\rm M_{\odot}\ yr^{-1})}'
labels['log10SFR_10'] = r'\log_{10}({\rm SFR_{10}}/{\rm M_{\odot}\ yr^{-1})}'

labels['log10Mstar_30'] = r'\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}'
labels['log10sSFR'] = r'\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}'
labels['log10sSFR_inst'] = r'\log_{10}({\rm sSFR_{inst}}/{\rm Gyr^{-1})}'
labels['log10sSFR_10'] = r'\log_{10}({\rm sSFR_{10}}/{\rm Gyr^{-1})}'

labels['age'] = r'age/{\rm Myr}'
labels['log10age'] = log10x(labels['age'])

labels['timescale'] = r'SF\ timescale\ \tau_{SF}/{\rm Myr}'


labels['Z'] = r'Z'
labels['log10Z'] = log10x(labels['Z'])



labels['log10SFRinst/10'] = r'log_{10}(SFR_{inst}/SFR_{10})'
labels['log10SFR10/50'] = r'log_{10}(SFR_{10}/SFR_{50})'
labels['log10SFR50/200'] = r'log_{10}(SFR_{50}/SFR_{200})'

labels['R'] = labels['log10SFR50/200']
labels['R2'] = labels['log10SFR10/50']

labels['lambda'] = r'\lambda/Myr^{-1}'
labels['log10lambda'] = rf'log_{{10}}{labels["lambda"]}'

labels['skew'] = r'skew'
labels['nvariance'] = r'(\lambda\sigma)^2'
labels['nkurtosis'] = r'excess\ kurtosis'

for f in ['FUV','NUV','V']:
    labels[f'log10{f}'] = rf'\log_{{10}}(L_{{ {f} }}/erg\ s^{{-1}}\ Hz^{{-1}})'
    labels[f'log10{f}_DustModelI'] = labels['log10FUV']
    labels[f'log10{f}_Intrinsic'] = rf'\log_{{10}}(L_{{ {f} }}^{{intrinsic}}/erg\ s^{{-1}}\ Hz^{{-1}})'



labels['AFUV'] = r'A_{FUV}'
labels['log10BH_Mass'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot})}'
labels['log10BH_Mdot'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot}\ yr^{-1})}'
# labels['log10VMTOL'] = labels['log10M*/LV'] = r'\log_{10}[({\rm M_{\star}}/L_{V})/({\rm M_{\odot}/erg\ s^{-1}\ Hz^{-1})]'
labels['log10VMTOL'] = labels['log10M*/LV'] = r'\log_{10}({\rm M_{\star}}/L_{V})'


labels['BD'] = r'L_{H\alpha}/L_{H\beta}'
labels['log10BD'] = log10x(labels['BD'])







# default plot limits

limits = {}
limits['log10Mstar_30'] = [8.5,11.49]

sfr_limits = [0.,3]
limits['log10SFR_10'] = sfr_limits
limits['log10SFR_inst_30'] = sfr_limits

limits['log10BH_Mass'] = [5.1,9.9]
limits['log10BH_Mdot'] = [-2.9,1.9]

limits['log10Z'] = [-3.19,-1.51]

limits['age'] = [0,400]
limits['log10age'] = [1.4, 2.9]
limits['timescale'] = [0,500]


limits['log10sSFR'] = [-0.9,1.9]
limits['log10sSFR_inst'] = limits['log10sSFR']
limits['log10sSFR_10'] = limits['log10sSFR']

limits['log10HbetaEW'] = [0.01,2.4]
limits['AFUV'] = [0,3.9]
limits['log10FUV'] = [28.1,30.1]

limits['log10M*/LV'] = limits['log10VMTOL'] = [-20.4, -19.1]

limits['BD'] = [2, 5]
limits['log10BD'] = [0.,1.5]

limits['beta'] = [-2.7,-1.1]
limits['log10BB'] = [-0.09,0.49]
limits['log10HbetaEW'] = [0.01,2.4]

limits['beta_Intrinsic'] = limits['beta_Pure_Stellar'] = [-2.9, -1.7]
limits['log10BB_Intrinsic'] = limits['log10BB_Pure_Stellar'] = limits['log10BB']
limits['log10HbetaEW_Intrinsic'] = [0.01,2.4]


limits['log10SFR50/200'] = [-0.9,0.9]
limits['log10SFR10/50'] = [-0.9,0.9]
limits['log10SFRinst/10'] = [-0.9,0.9]


limits['log10lambda'] = [0.21, 1.49]
limits['nvariance'] = [0., 2.9]
limits['skew'] = [-1.9, 3.9]
limits['nkurtosis'] = [-1.9, 19]

# default scalings to the units above

scalings = {}
scalings['Mstar_30'] = 1E10
scalings['BH_Mass'] = 1E10
scalings['age'] = 1000 # Gyr -> Myr

# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
BH_Mdot_scaling = h * 6.445909132449984E23  # g/s
BH_Mdot_scaling /= constants.M_sun.to('g').value  # convert to M_sol/s
BH_Mdot_scaling *= units.yr.to('s')  # convert to M_sol/yr
scalings['BH_Mdot'] = BH_Mdot_scaling


# bins

nbins = {}
nbins['log10Mstar_30'] = 30








def explore(fl):
    with h5py.File(fl.fname,'r') as f:
        f.visit(print)





def get_datasets(fl, tag, quantities, apply_scalings = True):

    halo = fl.halos


    # --- create dictionary of quantities

    Q = {}
    d = {}
    D = {}
    for q in quantities:

        if not q['name']:
            qid = q['dataset']
        else:
            qid = q['name']

        d[qid] = fl.load_dataset(q['dataset'], arr_type=q['path'])
        D[qid] = np.array([])
        Q[qid] = q


    # --- read in weights
    df = pd.read_csv('/cosma/home/dp004/dc-wilk2/data/flare/modules/flares/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    D['weight'] = np.array([])
    D['delta'] = np.array([])

    for ii in range(len(halo)):
        for qid in Q.keys():
            D[qid] = np.append(D[qid], d[qid][halo[ii]][tag])
        D['weight'] = np.append(D['weight'], np.ones(np.shape(d[qid][halo[ii]][tag]))*weights[ii])
        D['delta'] = np.append(D['delta'], np.ones(np.shape(d[qid][halo[ii]][tag]))*deltas[ii])

    # --- apply standard scaling
    if apply_scalings:
        for qid, q in Q.items():
            if qid in scalings:
                D[qid] *= scalings[qid]

    # --- create logged versions of the quantities
    for qid, q in Q.items():
        if q['log10']:
            D['log10'+qid] = np.log10(D[qid])

    return D



def get_particle_datasets(fl, tag):

    sims = fl.halos

    Mstar_30 = fl.load_dataset('Mstar_30', arr_type='Galaxy')
    S_length = fl.load_dataset('S_Length', arr_type='Galaxy')

    df = pd.read_csv('/cosma/home/dp004/dc-wilk2/data/flare/modules/flares/weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    D = {}

    D['Mstar_30'] = np.array([])
    D['weight'] = np.array([])
    D['delta'] = np.array([])

    ks = ['Age', 'Mass', 'MassInitial', 'Z']

    for k in ks:
        D[k] = []

    for ii in range(len(sims)):

        D['Mstar_30'] = np.append(D['Mstar_30'], Mstar_30[sims[ii]][tag])
        D['weight'] = np.append(D['weight'], np.ones(np.shape(Mstar_30[sims[ii]][tag]))*weights[ii])
        D['delta'] = np.append(D['delta'], np.ones(np.shape(Mstar_30[sims[ii]][tag]))*deltas[ii])

        for k in ks:
            D[k] += fl.get_particles('S_'+k, S_length[fl.halos[ii]][tag], halo = sims[ii], tag = tag, return_dict = False)




    D['Mstar_30'] *= 1E10
    D['log10Mstar_30'] = np.log10(D['Mstar_30'])

    for k in ks:
        D[k] = np.array(D[k])

    D['pweight'] = []
    for v,w in zip(D[ks[0]],D['weight']):
        D['pweight'].append(w*np.ones(v.shape))
    D['pweight'] = np.array(D['pweight'])


    return D








def simple_wcbar(D, x, y, z, s = None, labels = labels, limits = limits,  cmap = cm.viridis, add_weighted_median = True):

    #
    # # --- if no selection provided select all galaxies
    # if not s:
    #     s = D[x] == D[x]

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y, z]:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in [x, y, z]:
        if v not in labels.keys():
            labels[v] = v


    # --- get template figure from flare.plt
    fig, ax, cax = fplt.simple_wcbar()

    # --- define colour scale
    norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])


    # --- plot
    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[z][s])))

    # --- weighted median Lines

    if add_weighted_median:
        bins = np.linspace(*limits[x], 20)
        bincen = (bins[:-1]+bins[1:])/2.
        out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

        ax.plot(bincen, out[:,1], c='k', ls = '-')
        # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)


    ax.set_xlim(limits[x])
    ax.set_ylim(limits[y])

    ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 9)


    # --- add colourbar

    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])
    cbar = fig.colorbar(cmapper, cax=cax, orientation='vertical')
    cbar.set_label(rf'$\rm {labels[z]} $')

    return fig, ax, cax





def simple_wcbar_connected(D, x, y1, y2, z, s = None, labels = labels, limits = limits,  cmap = cm.viridis):

    """ Same as above but you provide two quantities (e.g. intrinsic and dust attenuated beta) to compare the effect """

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y1, y2, z]:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in [x, y1, y2, z]:
        if v not in labels.keys():
            labels[v] = v


    # --- get template figure from flare.plt
    fig, ax, cax = fplt.simple_wcbar()

    # --- define colour scale
    norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])


    # --- plot

    for x_, y1_, y2_, z_ in zip(D[x][s],D[y1][s],D[y2][s],D[z][s]):
        ax.plot([x_]*2, [y1_, y2_], c=cmap(norm(z_)), lw=1)


    # print(limits[x])
    #
    # ax.set_xlim(limits[x])
    # ax.set_ylim(limits[y1])

    ax.set_ylabel(rf'$\rm {labels[y1]}$', fontsize = 9)
    ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 9)


    # --- add colourbar

    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])
    cbar = fig.colorbar(cmapper, cax=cax, orientation='vertical')
    cbar.set_label(rf'$\rm {labels[z]} $')

    return fig, ax, cax










def simple_wcbar_whist(D, x, y, z, s = None, labels = labels, limits = limits,  cmap = cm.viridis, add_weighted_median = True, base_size = 3.5):

    left  = 0.15
    height = 0.70
    bottom = 0.15
    width = 0.6
    hwidth = 0.15


    fig = plt.figure(figsize = (base_size, base_size*width/height))

    ax = fig.add_axes((left, bottom, width, height))
    hax = fig.add_axes([left+width, bottom, hwidth, height])
    cax = fig.add_axes([left, bottom+height, width, 0.03])


    #
    # # --- if no selection provided select all galaxies
    # if not s:
    #     s = D[x] == D[x]

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y, z]:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in [x, y, z]:
        if v not in labels.keys():
            labels[v] = v




    # --- define colour scale
    norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])


    # --- plot
    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[z][s])))


    # --- add histogram
    bins = np.linspace(*limits[y], 20)
    bincen = (bins[:-1]+bins[1:])/2.
    H, bin_edges = np.histogram(D[y][s], bins = bins, range = limits[x], density = True)
    Hw, bin_edges = np.histogram(D[y][s], bins = bins, range = limits[x], weights = D['weight'][s], density = True)


    X = []
    Y = []
    bef = 0.0
    for i,be in enumerate(bin_edges[:-1]):
        X.append(be)
        Y.append(bef)
        X.append(be)
        Y.append(Hw[i])
        bef = Hw[i]

    hax.plot(Y, X, c='k', ls = '-', lw=1)

    # hax.plot(H, bincen, c='k', ls = ':', lw=1)


    # hax.plot(H, bincen, c='k', ls = ':', lw=1)
    # hax.plot(Hw, bincen, c='k', ls = '-', lw=1)



    hax.set_xlim([0,1.2*np.max(Y)])
    hax.set_xticks([])
    hax.set_yticks([])

    # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)



    # --- weighted median Lines

    if add_weighted_median:
        bins = np.linspace(*limits[x], 20)
        bincen = (bins[:-1]+bins[1:])/2.
        out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])
        ax.plot(bincen, out[:,1], c='k', ls = '-')
        # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)


    ax.set_xlim(limits[x])
    ax.set_ylim(limits[y])

    ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 9)


    # --- add colourbar

    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])
    cbar = fig.colorbar(cmapper, cax=cax, orientation='horizontal')
    cbar.set_label(rf'$\rm {labels[z]} $')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    return fig, ax, cax, hax






def simple(D, x, y, s = None, labels = labels, limits = limits, nbins = nbins, add_weighted_median = True):

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y]:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in [x, y]:
        if v not in labels.keys():
            labels[v] = v

    # --- if no bins provided just use the name
    for v in [x, y]:
        if v not in nbins.keys():
            nbins[v] = 25




    # --- get template figure from flare.plt
    fig, ax = fplt.simple()

    # --- plot
    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

    # --- weighted median Lines

    if add_weighted_median:
        bins = np.linspace(*limits[x], nbins[x])
        bincen = (bins[:-1]+bins[1:])/2.
        out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

        ax.plot(bincen, out[:,1], c='k', ls = '-')
        # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)


    ax.set_xlim(limits[x])
    ax.set_ylim(limits[y])

    ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 9)

    return fig






def simple_zevo(Dt, x, y, s = None, labels = labels, limits = limits):

    """ redshift evolution of a quantity """

    z = list(Dt.keys())[-1]

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y]:
        if v not in limits.keys():
            limits[v] = [np.min(D[z][v][s[-1]]), np.max(D[z][v][s[-1]])]

    # --- if no labels provided just use the name
    for v in [x, y]:
        if v not in labels.keys():
            labels[v] = v


    # --- get template figure from flare.plt
    fig, ax = fplt.simple()

    norm = mpl.colors.Normalize(vmin=5, vmax=10)
    cmap = cm.plasma


    for z, D in Dt.items():

        c = cmap(norm(z))

        bins = np.linspace(*limits[x], 20)
        bincen = (bins[:-1]+bins[1:])/2.
        out = flares.binned_weighted_quantile(D[x][s[z]],D[y][s[z]], D['weight'][s[z]],bins,[0.84,0.50,0.16])
        N, be = np.histogram(D['log10Mstar_30'][s[z]], bins=bins)
        # print(len(N), len(out[:,1]))

        s2 = N>10

        ax.plot(bincen, out[:,1], c=c, ls = ':')
        ax.plot(bincen[s2], out[:,1][s2], c=c, ls = '-', label = rf'$\rm z={z}$')
        # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)


    ax.set_xlim(limits[x])
    ax.set_ylim(limits[y])

    ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 9)

    ax.legend(fontsize=7)

    return fig, ax







def linear(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50, full_width = True):


    # --- if no limits provided base limits on selected data ranges
    for v in properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in properties:
        if v not in labels.keys():
            labels[v] = v

    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap

    N = len(properties)-1

    if full_width:
        left = 0.1
        top = 0.9
        bottom = 0.2
        right = 0.9
    else:
        left = 0.15
        top = 0.9
        bottom = 0.2
        right = 0.85

    panel_width = (right-left)/N
    panel_height = top-bottom

    if full_width:
        fig, axes = plt.subplots(1, N, figsize = (7,7/(panel_height/panel_width)), sharey = True)
    else:
        fig, axes = plt.subplots(1, N, figsize = (3.5,3.5/(panel_height/panel_width)), sharey = True)

    plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)


    y = properties[0]

    for ax, x in zip(axes, properties[1:]):


        # --- scatter plot here

        if scatter_colour_quantity:
            ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[scatter_colour_quantity][s])))
        else:
            ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

        # --- weighted median Lines

        bins = np.linspace(*limits[x], 20)
        bincen = (bins[:-1]+bins[1:])/2.
        out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

        ax.plot(bincen, out[:,1], c='k', ls = '-')
        # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

        ax.set_xlim(limits[x])
        ax.set_ylim(limits[y])

        ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 8)



    axes[0].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 8)

    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        cax = fig.add_axes([right, bottom, 0.015, top-bottom])
        fig.colorbar(cmapper, cax=cax, orientation='vertical')
        cax.set_ylabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 7)
        cax.tick_params(axis='y', labelsize=6)

    return fig







def linear_mcol(D, diagnostics, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50, full_width = True):


    # --- if no limits provided base limits on selected data ranges
    for v in properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in properties:
        if v not in labels.keys():
            labels[v] = v

    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap


    Ny = len(diagnostics)
    Nx = len(properties)

    if full_width:
        left = 0.1
        top = 0.9
        bottom = 0.1
        right = 0.9
    else:
        left = 0.15
        top = 0.95
        bottom = 0.125
        right = 0.85

    panel_width = (right-left)/Nx
    panel_height = (top-bottom)/Ny

    if full_width:
        fig, axes = plt.subplots(Ny, Nx, figsize = (7,7/(panel_height/panel_width)), sharey = 'row')
    else:
        fig, axes = plt.subplots(Ny, Nx, figsize = (3.5,3.5/(panel_height/panel_width)), sharey = 'row')

    plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)


    for j,y in enumerate(diagnostics):

        for i,x in enumerate(properties):

            ax = axes[j, i]

            # --- scatter plot here


            if scatter_colour_quantity:
                ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[scatter_colour_quantity][s])))
            else:
                ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

            # --- weighted median Lines

            bins = np.linspace(*limits[x], 20)
            bincen = (bins[:-1]+bins[1:])/2.
            out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

            ax.plot(bincen, out[:,1], c='k', ls = '-')
            # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

            ax.set_xlim(limits[x])
            ax.set_ylim(limits[y])

            ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 8)

            if i==0: ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 8)


    # axes[0].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 8)

    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        cax = fig.add_axes([right, bottom, 0.015, top-bottom])
        fig.colorbar(cmapper, cax=cax, orientation='vertical')
        cax.set_ylabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 7)
        cax.tick_params(axis='y', labelsize=6)

    return fig













def linear_redshift(D, zeds, x, y, s, labels = labels, limits = limits, nbins = nbins, scatter = True, scatter_colour_quantity = False, scatter_cmap = None, bins = 50, rows = 1, add_weighted_median = True, add_weighted_range = False):


    # --- if no limits provided base limits on selected data ranges
    for v in [x,y]:
        if v not in limits.keys():

            limits[v] = [np.min(D[zeds[-1]][v][s[zeds[-1]]]), np.max(D[zeds[-1]][v][s[zeds[-1]]])]

    # --- if no labels provided just use the name
    for v in [x,y]:
        if v not in labels.keys():
            labels[v] = v

    # --- if no bins provided just use the name
    for v in [x, y]:
        if v not in nbins.keys():
            nbins[v] = 25


    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap

    N = len(zeds)


    if rows == 1:
        left = 0.1
        top = 0.9
        bottom = 0.25
        right = 0.9
        panel_width = (right-left)/N
        panel_height = top-bottom
        fig, axes = plt.subplots(1, N, figsize = (7,7/(panel_height/panel_width)), sharey = True)
        plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)
    if rows == 2:
        left = 0.1
        top = 0.95
        bottom = 0.1
        right = 0.9
        panel_width = (right-left)/int(N/2)
        panel_height = top-bottom
        fig, axes = plt.subplots(2, int(N/2), figsize = (7,rows*7/(panel_height/panel_width)), sharey = True, sharex = True)
        plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)
        axes = axes.flatten()





    for ax, z in zip(axes, zeds):

        # --- scatter plot here
        if scatter:
            if scatter_colour_quantity:
                ax.scatter(D[z][x][s[z]],D[z][y][s[z]], s=1, alpha=0.5, c = cmap(norm(D[z][scatter_colour_quantity][s[z]])))
            else:
                ax.scatter(D[z][x][s[z]],D[z][y][s[z]], s=1, alpha=0.5, c = 'k')

        # --- weighted median Lines

        if add_weighted_median:
            bins = np.linspace(*limits[x], nbins[x])
            bincen = (bins[:-1]+bins[1:])/2.
            out = flares.binned_weighted_quantile(D[z][x][s[z]],D[z][y][s[z]], D[z]['weight'][s[z]],bins,[0.84,0.50,0.16])

            N, bin_edges = np.histogram(D[z][x][s[z]], bins=bins)


            i = np.array(range(len(N)))

            ss = i[N<1]
            if len(ss)>0:
                sN = i[i<ss[0]]
            else:
                sN = i

            ax.plot(bincen[sN], out[:,1][sN], c='k', ls = '--')

            ss = i[N<10]
            if len(ss)>0:
                sN = i[i<ss[0]]
            else:
                sN = i


            ax.plot(bincen[sN], out[:,1][sN], c='k', ls = '-')

            if add_weighted_range:
                ax.fill_between(bincen[sN], out[:,0][sN], out[:,2][sN], color='k', alpha = 0.2)

        ax.set_xlim(limits[x])
        ax.set_ylim(limits[y])

        if rows == 1:
            ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
        if rows == 2:
            ax.text(0.5, 1.01, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 8)



    axes[0].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    if rows == 2: axes[3].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)

    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        cax = fig.add_axes([right, bottom, 0.015, top-bottom])
        fig.colorbar(cmapper, cax=cax, orientation='vertical')
        cax.set_ylabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 7)
        if rows==2: cax.set_ylabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 8)
        cax.tick_params(axis='y', labelsize=6)


    fig.text(left+(right-left)/2, 0.04, rf'$\rm {labels[x]}$', ha='center', fontsize = 9)


    return fig, axes








def linear_redshift_density(D, zeds, x, y, s, labels = labels, limits = limits, rows = 1):


    # --- if no limits provided base limits on selected data ranges
    for v in [x,y]:
        if v not in limits.keys():
            limits[v] = [np.min(D[zeds[-1]][v][s[zeds[-1]]]), np.max(D[zeds[-1]][v][s[zeds[-1]]])]

    # --- if no labels provided just use the name
    for v in [x,y]:
        if v not in labels.keys():
            labels[v] = v


    N = len(zeds)

    if rows == 1:
        left = 0.1
        top = 0.9
        bottom = 0.25
        right = 0.9
        panel_width = (right-left)/N
        panel_height = top-bottom
        fig, axes = plt.subplots(1, N, figsize = (7,7/(panel_height/panel_width)), sharey = True)
        plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)
    if rows == 2:
        left = 0.1
        top = 0.95
        bottom = 0.1
        right = 0.9
        panel_width = (right-left)/int(N/2)
        panel_height = top-bottom
        fig, axes = plt.subplots(2, int(N/2), figsize = (7,rows*7/(panel_height/panel_width)), sharey = True, sharex = True)
        plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)
        axes = axes.flatten()


    ldelta_bins = np.linspace(-0.25,0.25,6)

    norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
    cmap = cm.Spectral_r



    for ax, z in zip(axes, zeds):

        for ldelta in ldelta_bins:

            sd = s[z]&(np.fabs(D[z]['ldelta']-ldelta)<0.05)

            # --- weighted median Lines

            bins = np.linspace(*limits[x], 20)
            bincen = (bins[:-1]+bins[1:])/2.
            out = flares.binned_weighted_quantile(D[z][x][sd],D[z][y][sd], D[z]['weight'][sd],bins,[0.84,0.50,0.16])


            # ax.plot(bincen, out[:,1], c=cmap(norm(ldelta)), ls = '-', label = rf'$\rm {ldelta-0.05:.2f}<\log_{{10}}(1+\delta)<{ldelta+0.05:.2f}$')
            ax.plot(bincen, out[:,1], c=cmap(norm(ldelta)), ls = '-', label = rf'$\rm [{ldelta-0.05:.2f},{ldelta+0.05:.2f})$')
            # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

        ax.set_xlim(limits[x])
        ax.set_ylim(limits[y])

        if rows == 1:
            ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
        if rows == 2:
            ax.text(0.5, 1.01, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 8)



    axes[0].legend(title = r'$\rm \log_{{10}}(1+\delta)$', title_fontsize = 6, fontsize = 5, labelspacing = 0.0)

    axes[0].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)
    if rows == 2: axes[3].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)

    fig.text(left+(right-left)/2, 0.04, rf'$\rm {labels[x]}$', ha='center', fontsize = 9)


    return fig












def linear_redshift_mcol(D, zeds, x, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50, add_linear_fit = False, height = 1):


    # --- if no limits provided base limits on selected data ranges
    for v in [x]+properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[zeds[-1]][v][s[zeds[-1]]]), np.max(D[zeds[-1]][v][s[zeds[-1]]])]

    # --- if no labels provided just use the name
    for v in [x]+properties:
        if v not in labels.keys():
            labels[v] = v

    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap

    Np = len(properties)
    N = len(zeds)

    left = 0.1
    top = 0.9
    bottom = 0.15 #somewhat dependent on Np
    right = 0.9

    panel_width = (right-left)/N
    panel_height = (top-bottom)/Np

    fig, axes = plt.subplots(Np, N, figsize = (7,height*7/(panel_height/panel_width)), sharex = True, sharey = 'row')
    plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)


    for j,y in enumerate(properties):

        for i,z in enumerate(zeds):

            ax = axes[j, i]

            # --- scatter plot here

            if scatter_colour_quantity:
                ax.scatter(D[z][x][s[z]],D[z][y][s[z]], s=1, alpha=0.5, c = cmap(norm(D[z][scatter_colour_quantity][s[z]])))
            else:
                ax.scatter(D[z][x][s[z]],D[z][y][s[z]], s=1, alpha=0.5, c = 'k')

            # --- weighted median Lines

            bins = np.linspace(*limits[x], 20)
            bincen = (bins[:-1]+bins[1:])/2.
            out = flares.binned_weighted_quantile(D[z][x][s[z]],D[z][y][s[z]], D[z]['weight'][s[z]],bins,[0.84,0.50,0.16])

            ax.plot(bincen, out[:,1], c='k', ls = '-')
            # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

            # --- linear first

            if add_linear_fit:

                # fit_p = np.polyfit(D[z][x][s[z]],D[z][y][s[z]], 1, w = D[z]['weight'][s[z]])

                x_ = D[z][x][s[z]]
                y_ = D[z][y][s[z]]
                w_ = D[z]['weight'][s[z]]

                s_ = (~np.isnan(x_))&(~np.isnan(y_))&(~np.isinf(y_)) # capture NaNs

                fit_p = np.polyfit(x_[s_],y_[s_], 1, w = w_[s_])
                fit = lambda x: fit_p[1] + fit_p[0]*x
                ax.plot(limits[x], fit(np.array(limits[x])), lw=1, c='k', ls='--')

            ax.set_xlim(limits[x])
            ax.set_ylim(limits[y])

        axes[j, 0].set_ylabel(rf'$\rm {labels[y]}$', fontsize = 9)

    for i,z in enumerate(zeds):
        axes[0, i].text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=axes[0, i].transAxes, fontsize = 7)



    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        cax = fig.add_axes([right, bottom, 0.015, top-bottom])
        fig.colorbar(cmapper, cax=cax, orientation='vertical')
        cax.set_ylabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 7)
        cax.tick_params(axis='y', labelsize=6)


    fig.text(left+(right-left)/2, 0.04, rf'$\rm {labels[x]}$', ha='center', fontsize = 9)

    return fig, axes



























def corner_plot(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50):
    return corner_whist(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = scatter_colour_quantity, scatter_cmap = scatter_cmap, bins = bins)




def corner_whist(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50):


    # --- if no limits provided base limits on selected data ranges
    for v in properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in properties:
        if v not in labels.keys():
            labels[v] = v

    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap

    N = len(properties)

    fig, axes = plt.subplots(N, N, figsize = (7,7))
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

    for i in np.arange(N):
        for j in np.arange(N):
            axes[i, j].set_axis_off()

    for i,x in enumerate(properties):
        for j,y in enumerate(properties[1:][::-1]):

            jj = N-1-j
            ii = i

            ax = axes[jj, ii]

            if j+i<(N-1):
                ax.set_axis_on()

                # --- scatter plot here

                if scatter_colour_quantity:
                    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[scatter_colour_quantity][s])))
                else:
                    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

                # --- weighted median Lines

                bins = np.linspace(*limits[x], 20)
                bincen = (bins[:-1]+bins[1:])/2.
                out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

                ax.plot(bincen, out[:,1], c='k', ls = '-')
                # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

                ax.set_xlim(limits[x])
                ax.set_ylim(limits[y])

            if i == 0: # first column
                ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 7)
            else:
                ax.yaxis.set_ticklabels([])

            if j == 0: # first row
                ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 7)
            else:
                ax.xaxis.set_ticklabels([])

            # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)


        # --- histograms

        ax = axes[ii, ii]
        ax.set_axis_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        X = D[x][s]

        H, bin_edges = np.histogram(X, bins = bins, range = limits[x])
        Hw, bin_edges = np.histogram(X, bins = bins, range = limits[x], weights = D['weight'][s])

        Hw *= np.max(H)/np.max(Hw)

        bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5


        ax.fill_between(bin_centres, H*0.0, H, color='0.9')
        ax.plot(bin_centres, Hw, c='0.7', lw=1)

        ax.set_ylim([0.0,np.max(H)*1.2])


    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        cax = fig.add_axes([0.25, 0.87, 0.5, 0.015])
        fig.colorbar(cmapper, cax=cax, orientation='horizontal')
        cax.set_xlabel(rf'$\rm {labels[scatter_colour_quantity]} $')

    return fig







def corner(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50, full_width = True):


    # --- if no limits provided base limits on selected data ranges
    for v in properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in properties:
        if v not in labels.keys():
            labels[v] = v

    if scatter_colour_quantity:
        norm = mpl.colors.Normalize(vmin=limits[scatter_colour_quantity][0], vmax=limits[scatter_colour_quantity][1])
        cmap = scatter_cmap

    N = len(properties)-1


    if full_width:
        fig, axes = plt.subplots(N, N, figsize = (7,7))
    else:
        fig, axes = plt.subplots(N, N, figsize = (3.5,3.5))


    if full_width:
        left = 0.1
        right = 0.9
        bottom = 0.1
        top = 0.9
    else:
        left = 0.15
        right = 0.95
        bottom = 0.1
        top = 0.9


    plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)

    for i in np.arange(N):
        for j in np.arange(N):
            axes[i, j].set_axis_off()

    for i,x in enumerate(properties[:-1]):
        for j,y in enumerate(properties[1:][::-1]):

            jj = N-1-j
            ii = i

            ax = axes[jj, ii]

            if j+i<N:
                ax.set_axis_on()

                # --- scatter plot here

                if scatter_colour_quantity:
                    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[scatter_colour_quantity][s])))
                else:
                    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

                # --- weighted median Lines

                bins = np.linspace(*limits[x], 20)
                bincen = (bins[:-1]+bins[1:])/2.
                out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

                ax.plot(bincen, out[:,1], c='k', ls = '-')
                # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

                # ax.text(0.5, 0.5, f'{ii}_{jj}',transform=ax.transAxes)

                ax.set_xlim(limits[x])
                ax.set_ylim(limits[y])

            if i == 0: # first column
                ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 7)
            else:
                ax.yaxis.set_ticklabels([])

            if j == 0: # first row
                ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 7)
            else:
                ax.xaxis.set_ticklabels([])

            # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)

    # --- add colourbar

    if scatter_colour_quantity:

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        width = right-left
        height = top - bottom

        cax = fig.add_axes([left+width/2 + 0.025, bottom+height/2 + 0.2, width/2 - 0.025, 0.025])
        fig.colorbar(cmapper, cax=cax, orientation='horizontal')
        cax.set_xlabel(rf'$\rm {labels[scatter_colour_quantity]} $', fontsize = 6)
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.tick_params(axis='x', labelsize=5)

    return fig







def corner3(D, properties, s, cmaps, labels = labels, limits = limits, bins = 50, full_width = True):


    # --- if no limits provided base limits on selected data ranges
    for v in properties:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in properties:
        if v not in labels.keys():
            labels[v] = v

    N = len(properties)-1


    if full_width:
        left = 0.1
        right = 0.9
        bottom = 0.1
        top = 0.9
        fig, axes = plt.subplots(N, N, figsize = (7,7))
    else:
        left = 0.15
        right = 0.95
        bottom = 0.1
        top = 0.9
        fig, axes = plt.subplots(N, N, figsize = (3.5,3.5))


    plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)

    for i in np.arange(N):
        for j in np.arange(N):
            axes[i, j].set_axis_off()

    for i,x in enumerate(properties[:-1]):
        for j,y in enumerate(properties[1:][::-1]):

            jj = N-1-j
            ii = i


            ax = axes[jj, ii]

            if j+i<N:
                ax.set_axis_on()

                z = list(set(properties) ^ set([x, y]))[0]

                norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])
                cmap = cmaps[z]


                # --- scatter plot here

                ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[z][s])))


                # --- weighted median Lines

                bins = np.linspace(*limits[x], 20)
                bincen = (bins[:-1]+bins[1:])/2.
                out = flares.binned_weighted_quantile(D[x][s],D[y][s], D['weight'][s],bins,[0.84,0.50,0.16])

                ax.plot(bincen, out[:,1], c='k', ls = '-')
                # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

                # ax.text(0.5, 0.5, f'{ii}_{jj}',transform=ax.transAxes)

                ax.set_xlim(limits[x])
                ax.set_ylim(limits[y])

            if i == 0: # first column
                ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 7)
            else:
                ax.yaxis.set_ticklabels([])

            if j == 0: # first row
                ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 7)
            else:
                ax.xaxis.set_ticklabels([])

            # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)

    # --- add colourbar


    for i, z in enumerate(properties):

        norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])
        cmap = cmaps[z]

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        width = right - left
        height = top - bottom

        cax = fig.add_axes([left+width/2+ 0.025, bottom+height/2 + i*width/(2*len(properties)) + 0.025, width/2 - 0.025, 0.015])
        fig.colorbar(cmapper, cax=cax, orientation='horizontal')
        cax.set_xlabel(rf'$\rm {labels[z]} $', fontsize = 6)
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.tick_params(axis='x', labelsize=5)

    return fig, axes
