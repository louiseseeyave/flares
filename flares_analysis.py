import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


import astropy.constants as constants
import astropy.units as units

import flares

import flare.plt as fplt



fancy = lambda x: r'$\rm '+x.replace(' ','\ ')+'$'
ml = lambda x: r'$\rm '+x+'$'


deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])


# default plot limits

labels = {}
labels['log10SFR_inst_30'] = r'\log_{10}({\rm SFR}/{\rm M_{\odot}\ yr^{-1})}'
labels['log10Mstar_30'] = r'\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}'
labels['log10sSFR'] = r'\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}'
labels['beta'] = r'\beta'
labels['log10HbetaEW'] = r'\log_{10}(H\beta\ EW/\AA)'
labels['log10FUV'] = r'\log_{10}(L_{FUV}/erg\ s^{-1}\ Hz^{-1})'
labels['AFUV'] = r'A_{FUV}'
labels['log10BH_Mass'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot})}'
labels['log10BH_Mdot'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot}\ yr^{-1})}'


# default plot limits

limits = {}
limits['log10Mstar_30'] = [8.0,11]
limits['log10BH_Mass'] = [5.1,9.9]
limits['log10BH_Mdot'] = [-2.9,1.9]
limits['beta'] = [-2.9,-1.1]
limits['log10sSFR'] = [-0.9,1.9]
limits['log10HbetaEW'] = [0.01,2.49]
limits['AFUV'] = [0,3.9]
limits['log10FUV'] = [28.1,29.9]


# default scalings to the units above

scalings = {}
scalings['Mstar_30'] = 1E10
scalings['BH_Mass'] = 1E10

# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
BH_Mdot_scaling = h * 6.445909132449984E23  # g/s
BH_Mdot_scaling /= constants.M_sun.to('g').value  # convert to M_sol/s
BH_Mdot_scaling *= units.yr.to('s')  # convert to M_sol/yr
scalings['BH_Mdot'] = BH_Mdot_scaling




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

    return fig



def simple_wcbar_whist(D, x, y, z, s = None, labels = labels, limits = limits,  cmap = cm.viridis, add_weighted_median = True):

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
    fig, ax, cax, hax = fplt.simple_wcbar_whist()

    # --- define colour scale
    norm = mpl.colors.Normalize(vmin=limits[z][0], vmax=limits[z][1])


    # --- plot
    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D[z][s])))


    # --- add histogram
    bins = np.linspace(*limits[y], 20)
    bincen = (bins[:-1]+bins[1:])/2.
    H, bin_edges = np.histogram(D[y][s], bins = bins, range = limits[x], density = True)
    Hw, bin_edges = np.histogram(D[y][s], bins = bins, range = limits[x], weights = D['weight'][s], density = True)

    hax.plot(H, bincen, c='k', ls = ':', lw=1)
    hax.plot(Hw, bincen, c='k', ls = '-', lw=1)
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
    cbar = fig.colorbar(cmapper, cax=cax, orientation='vertical')
    cbar.set_label(rf'$\rm {labels[z]} $')

    return fig, ax, cax, hax






def simple(D, x, y, s = None, labels = labels, limits = limits, add_weighted_median = True):

    # --- if no limits provided base limits on selected data ranges
    for v in [x, y]:
        if v not in limits.keys():
            limits[v] = [np.min(D[v][s]), np.max(D[v][s])]

    # --- if no labels provided just use the name
    for v in [x, y]:
        if v not in labels.keys():
            labels[v] = v


    # --- get template figure from flare.plt
    fig, ax = fplt.simple()

    # --- plot
    ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = 'k')

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

    return fig







def corner_plot(D, properties, s, labels = labels, limits = limits, scatter_colour_quantity = False, scatter_cmap = None, bins = 50):


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
