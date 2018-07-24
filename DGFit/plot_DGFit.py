#!/usr/bin/env python
#
# plot_DGFit
#
# Started: Dec 2015 (KDG)

from __future__ import print_function

import argparse

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib

from astropy.io import fits

from DGFit.ObsData import ObsData


def get_krange(x, logaxis=False, in_range=[0]):

    prange = np.array([0.0, 0.0])
    min_x = np.amin(x)
    max_x = np.amax(x)
    if logaxis:
        min_x = np.log10(min_x)
        max_x = np.log10(max_x)

    delta = max_x - min_x
    prange[0] = min_x - 0.1*delta
    prange[1] = max_x + 0.1*delta

    if logaxis:
        prange = np.power(10.0, prange)

    if len(in_range) > 1:
        prange[0] = np.minimum(prange[0], in_range[0])
        prange[1] = np.maximum(prange[1], in_range[1])

    return(prange)


# plot the size distributions
def plot_dgfit_sizedist(ax, hdulist, colors=['b', 'g'],
                        fontsize=12, multa4=True, plegend=True):

    if 'DISTPUNC' in hdulist[1].data.names:
        plot_uncs = True
    else:
        plot_uncs = False

    yrange = [0]
    for i in range(hdulist[0].header['NCOMPS']):
        hdu = hdulist[i+1]

        xvals = hdu.data['SIZE']*1e4
        yvals = hdu.data['DIST']
        if plot_uncs:
            yvals_punc = hdu.data['DISTPUNC']
            yvals_munc = hdu.data['DISTMUNC']

        if multa4:
            xvals4 = hdu.data['SIZE']**4
            yvals = yvals*xvals4
            if plot_uncs:
                yvals_punc = yvals_punc*xvals4
                yvals_munc = yvals_munc*xvals4

        yrange = get_krange(yvals, logaxis=True, in_range=yrange)
        if plot_uncs:
            yrange = get_krange(yvals-yvals_munc, logaxis=True,
                                in_range=yrange)
            yrange = get_krange(yvals+yvals_punc, logaxis=True,
                                in_range=yrange)

        ax.plot(xvals, yvals, colors[i]+'-', label=hdu.header['EXTNAME'])
        if plot_uncs:
            ax.errorbar(xvals, yvals, fmt=colors[i]+'o',
                        yerr=[yvals_munc, yvals_punc])

    if multa4:
        ylabel = r'$a^4 N_d(a)/N(HI)$'
    else:
        ylabel = r'$N_d(a)/N(HI)$'

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(get_krange(xvals, logaxis=True))
    ax.set_ylim(yrange)
    ax.set_xlabel(r'$a [\mu m]$', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if plegend:
        ax.legend()


# plot the atomic abundances
def plot_dgfit_abundances(ax, hdu, obsdata, colors=['r', 'b', 'g'],
                          fontsize=12):

    # plot the dust abundances
    atomnames = hdu.data['NAME']
    atomabund = hdu.data['ABUND']
    n_atoms = len(atomnames)
    aindxs = np.arange(n_atoms)
    width = 0.5
    ax.bar(aindxs+0.25*width, atomabund, width, color=colors[0])

    ax.errorbar(aindxs+0.75*width, [obsdata.abundance[x][0]
                                    for x in atomnames],
                yerr=[obsdata.abundance[x][1] for x in atomnames], fmt='ko')

    ax.set_ylabel(r'$N(X)/[10^6 N(HI)]$', fontsize=fontsize)
    ax.set_xticks(aindxs+(0.75*width))
    ax.set_xticklabels(atomnames)


# plot the extinction curves (total and components)
def plot_dgfit_extinction(ax, hdu, obsdata, colors=['r', 'b', 'g'],
                          fontsize=12, comps=True):

    yrange = [0.0, 0.0]

    ax.plot(hdu.data['WAVE'], hdu.data['EXT'], colors[0]+'-')
    yrange = get_krange(hdu.data['EXT'], logaxis=True)
    if comps:
        # linetypes = ['--', ':', '-.']
        linetypes = ['-', '-', '-', '-', '-']
        for i in range(len(hdu.data.names)-2):
            ax.plot(hdu.data['WAVE'], hdu.data['EXT'+str(i+1)],
                    colors[i+1]+linetypes[i])
            yrange = get_krange(hdu.data['EXT'+str(i+1)], logaxis=True,
                                in_range=yrange)

    ax.plot(obsdata.ext_waves, obsdata.ext_alnhi, 'k-', label='Observed')
    yrange = get_krange(obsdata.ext_alnhi, logaxis=True, in_range=yrange)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda [\mu m]$', fontsize=fontsize)
    ax.set_ylabel(r'$A(\lambda)/N(HI)$', fontsize=fontsize)

    ax.set_xlim(get_krange(hdu.data['WAVE'], logaxis=True))
    ax.set_ylim(yrange)


# plot the emission spectra (total and components)
def plot_dgfit_emission(ax, hdu, obsdata, colors=['r', 'b', 'g'],
                        fontsize=12, comps=True):

    yrange = [0.0, 0.0]

    ax.plot(hdu.data['WAVE'], hdu.data['EMIS'], colors[0]+'-')
    yrange = get_krange(hdu.data['EMIS'], logaxis=True)
    if comps:
        linetypes = ['-', '-', '-', '-', '-']
        for i in range(len(hdu.data.names)-2):
            ax.plot(hdu.data['WAVE'], hdu.data['EMIS'+str(i+1)],
                    colors[i+1]+linetypes[i])
            yrange = get_krange(hdu.data['EMIS'+str(i+1)], logaxis=True,
                                in_range=yrange)

    ax.errorbar(obsdata.ir_emission_waves, obsdata.ir_emission,
                yerr=obsdata.ir_emission_unc, fmt='ko', label='Observed')
    yrange = get_krange(obsdata.ir_emission, logaxis=True, in_range=yrange)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda [\mu m]$', fontsize=fontsize)
    ax.set_ylabel(r'$S$ $[MJy$ $sr^{-1}$ $N(HI)^{-1}]$', fontsize=fontsize)

    ax.set_xlim(get_krange(hdu.data['WAVE'], logaxis=True))
    ax.set_ylim(yrange)


# plot the dust scattering albedo
def plot_dgfit_albedo(ax, hdu, obsdata, colors=['r', 'b', 'g'],
                      fontsize=12, comps=True):

    yrange = [0.0, 0.0]

    ax.plot(hdu.data['WAVE'], hdu.data['ALBEDO'], colors[0]+'-')
    yrange = get_krange(hdu.data['ALBEDO'])
    if comps:
        linetypes = ['-', '-', '-', '-', '-']
        for i in range(len(hdu.data.names)-2):
            ax.plot(hdu.data['WAVE'], hdu.data['ALBEDO'+str(i+1)],
                    colors[i+1]+linetypes[i])
            yrange = get_krange(hdu.data['ALBEDO'+str(i+1)], in_range=yrange)

    ax.errorbar(obsdata.scat_a_waves, obsdata.scat_albedo,
                yerr=obsdata.scat_albedo_unc, fmt='ko', label='Observed')
    yrange = get_krange(obsdata.scat_albedo, in_range=yrange)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda [\mu m]$', fontsize=fontsize)
    ax.set_ylabel(r'$albedo$', fontsize=fontsize)

    ax.set_xlim(get_krange(hdu.data['WAVE'], logaxis=True))
    ax.set_ylim(yrange)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help=("file with the dust model details "
                              + "(size distribution, extinction, etc.)"))
    parser.add_argument("--smc", help="use an SMC sightline",
                        action="store_true")
    parser.add_argument("-p", "--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("-e", "--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("-pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()

    # setup the plots
    fontsize = 12
    font = {'size': fontsize}

    matplotlib.rc('font', **font)

    matplotlib.rc('lines', linewidth=2)
    matplotlib.rc('axes', linewidth=2)
    matplotlib.rc('xtick.major', width=2)
    matplotlib.rc('ytick.major', width=2)

    fig, ax = pyplot.subplots(ncols=3, nrows=2, figsize=(15, 10))

    # open the DGFit results
    hdulist = fits.open(args.filename)

    # get the observed data
    if args.smc:
        obsdata = ObsData('data_smc_azv215/azv215_50p_ext.fits',
                          'data_smc_azv215/azv215_avnhi.dat',
                          'data_smc_azv215/SMC_AzV215_abundances.dat',
                          None,
                          None)
    else:
        obsdata = ObsData(['data_mw_rv31/MW_diffuse_Gordon09_band_ext.dat',
                           'data_mw_rv31/MW_diffuse_Gordon09_iue_ext.dat',
                           'data_mw_rv31/MW_diffuse_Gordon09_fuse_ext.dat'],
                          'data_mw_rv31/MW_diffuse_Gordon09_avnhi.dat',
                          'data_mw_rv31/MW_diffuse_Jenkins09_abundances.dat',
                          'data_mw_rv31/MW_diffuse_Compiegne11_ir_emission.dat',
                          'dust_scat.dat',
                          ext_tags=['band', 'iue', 'fuse'])

    # plot the dust size distributions
    colors = ['b', 'g']
    plot_dgfit_sizedist(ax[0, 0], hdulist, fontsize=fontsize, multa4=False)
    plot_dgfit_sizedist(ax[1, 0], hdulist, fontsize=fontsize, plegend=False)

    # plot the abundances
    plot_dgfit_abundances(ax[0, 1], hdulist['ABUNDANCES'], obsdata,
                          fontsize=fontsize)

    # plot the resulting total and component extinction curves
    plot_dgfit_extinction(ax[1, 1], hdulist['EXTINCTION'], obsdata,
                          fontsize=fontsize)

    # plot the resulting total and component emission spectra
    if obsdata.fit_ir_emission:
        plot_dgfit_emission(ax[0, 2], hdulist['EMISSION'], obsdata,
                            fontsize=fontsize)

    # plot the resulting total and component emission spectra
    if obsdata.fit_scat_a:
        plot_dgfit_albedo(ax[1, 2], hdulist['ALBEDO'], obsdata,
                          fontsize=fontsize)

    pyplot.tight_layout()

    # show or save
    basename = args.filename
    basename.replace('.fits', '')
    if args.png:
        fig.savefig(basename+'.png')
    elif args.eps:
        fig.savefig(basename+'.eps')
    elif args.pdf:
        fig.savefig(basename+'.pdf')
    else:
        pyplot.show()
