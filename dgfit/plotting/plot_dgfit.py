import argparse

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
from astropy.io import fits

from dgfit.obsdata import ObsData


def get_krange(x, logaxis=False, in_range=[0]):
    prange = np.array([0.0, 0.0])
    if logaxis:
        gindxs = x > 0
        min_x = np.amin(x[gindxs])
        max_x = np.amax(x[gindxs])
    else:
        min_x = np.amin(x)
        max_x = np.amax(x)

    if logaxis:
        max_x = np.log10(max_x)
        if min_x <= 0.0:
            min_x = max_x - 10.0
        else:
            min_x = np.log10(min_x)

    delta = max_x - min_x
    prange[0] = min_x - 0.1 * delta
    prange[1] = max_x + 0.1 * delta

    if logaxis:
        prange = np.power(10.0, prange)

    if len(in_range) > 1:
        prange[0] = np.minimum(prange[0], in_range[0])
        prange[1] = np.maximum(prange[1], in_range[1])

    return prange


# plot the size distributions
def plot_dgfit_sizedist(
    ax,
    hdulist,
    colors=["C0", "C2", "C1", "C4", "C5", "C6"],
    fontsize=12,
    mass=True,
    plegend=True,
    ltype=["o-", "x-", "D-", "p-", "v-", "s-"],
    alpha=1.0,
    markers=1,
    file="none",
):
    if "DISTPUNC" in hdulist[1].data.names:
        plot_uncs = True
    else:
        plot_uncs = False

    yrange = [0]
    all_yvals = []
    j = 0
    for i in range(hdulist[0].header["NCOMPS"]):
        hdu = hdulist[i + 1]

        xvals = hdu.data["SIZE"] * 1e4
        yvals = hdu.data["DIST"]

        if np.sum(yvals) == 0:
            print(f"Composition {i} is zero")
            continue

        if plot_uncs:
            yvals_punc = hdu.data["DISTPUNC"]
            yvals_munc = hdu.data["DISTMUNC"]

        if mass:
            xvals3 = hdu.data["SIZE"] ** 3
            yvals = yvals * xvals3
            if plot_uncs:
                yvals_punc = yvals_punc * xvals3
                yvals_munc = yvals_munc * xvals3

        yrange = get_krange(yvals, logaxis=True, in_range=yrange)
        if plot_uncs:
            yrange = get_krange(yvals - yvals_munc, logaxis=True, in_range=yrange)
            yrange = get_krange(yvals + yvals_punc, logaxis=True, in_range=yrange)

        gindxs = yvals > 0
        all_yvals.append(np.max(yvals))

        ax.plot(
            xvals[gindxs],
            yvals[gindxs],
            colors[i] + ltype[i],
            markevery=markers,
            label=hdu.header["EXTNAME"],
            alpha=alpha,
        )
        if file != "none":
            length = len(xvals)
            uppers, lowers = np.loadtxt(file, unpack=True)
            comp_lower = lowers[j : j + length] * (hdu.data["SIZE"] ** 3)
            comp_upper = uppers[j : j + length] * (hdu.data["SIZE"] ** 3)
            ax.fill_between(
                xvals,
                comp_lower,
                comp_upper,
                color=colors[i],
                alpha=0.2,
                zorder=0,
            )
            j += length
            if j < len(lowers):
                if lowers[j] == 0.2:
                    j += 1
        if plot_uncs:
            ax.errorbar(
                xvals[gindxs],
                yvals[gindxs],
                fmt=colors[i] + "o",
                yerr=[yvals_munc[gindxs], yvals_punc[gindxs]],
                alpha=alpha,
            )

    if mass:
        ylabel = r"$m(a)/A(V)$"
    else:
        ylabel = r"$N_d(a)/A(V)$"

    ymax = max(all_yvals) * 100
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_ylim(1e-10, ymax)
    ax.set_xlabel(r"a $[\mu m]$", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if plegend:
        ax.legend()


# plot the atomic abundances
def plot_dgfit_abundances(
    ax, hdulist, obsdata, color="g", fontsize=12, plabel="None", plegend=False
):
    n_comps = hdulist[0].header["NCOMPS"]
    title = ""
    for i in range(n_comps):
        hdu = hdulist[i + 1]
        fluffy = hdu.header.get(f"F_a_{i}")
        if fluffy is not None:
            title += rf"$F_{{a {i + 1}}}$={fluffy:.2f}"
            if i != n_comps - 1:
                title += ", "
    hdu = hdulist["ABUNDANCES"]

    # plot the dust abundances
    atomnames = hdu.data["NAME"]
    atomabund = hdu.data["ABUND"]
    n_atoms = len(atomnames)
    aindxs = np.arange(n_atoms)
    width = 0.5
    ax.bar(
        aindxs + 0.75 * width, atomabund, width, color=color, alpha=0.15, label=plabel
    )

    if obsdata.obs_filenames["abund"] is not None:
        ax.errorbar(
            aindxs + 0.75 * width,
            [obsdata.abundance_av[x][0] for x in atomnames],
            yerr=[obsdata.abundance_av[x][1] for x in atomnames],
            fmt="ko",
            label="Observations",
        )

    ax.set_ylabel(r"$N(X)/A(V)$", fontsize=fontsize)
    ax.set_ylim(0)
    ax.set_xticks(aindxs + (0.75 * width))
    ax.set_xticklabels(atomnames)
    ax.set_title(title, fontsize=12)

    if plegend:
        ax.legend(loc="upper left")


# plot the extinction curves (total and components)
def plot_dgfit_extinction(
    ax,
    hdu,
    obsdata,
    colors=["C3", "C0", "C2", "C1", "C4", "C5", "C6"],
    fontsize=12,
    comps=True,
    ltype="-",
):
    ax.plot(hdu.data["WAVE"], hdu.data["EXT"], colors[0] + ltype)
    yrange = get_krange(hdu.data["EXT"], logaxis=True)
    if comps:
        # linetypes = ['--', ':', '-.']
        # linetypes = ["-", "-", "-", "-", "-"]
        for i in range(len(hdu.data.names) - 2):

            if np.sum(hdu.data["EXT" + str(i + 1)]) == 0:
                continue

            ax.plot(
                hdu.data["WAVE"],
                hdu.data["EXT" + str(i + 1)],
                colors[i + 1] + ltype,
            )
            yrange = get_krange(
                hdu.data["EXT" + str(i + 1)], logaxis=True, in_range=yrange
            )

    if obsdata.obs_filenames["ext"] is not None:
        ax.plot(obsdata.ext_waves, obsdata.ext_alav, "k-", label="Observed")
        yrange_obs = get_krange(obsdata.ext_alav, logaxis=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)
    ax.set_ylabel(r"$A(\lambda)/A(V)$", fontsize=fontsize)
    ax.set_xlim(get_krange(hdu.data["WAVE"], logaxis=True))
    ax.set_ylim(yrange_obs)


# plot the emission spectra (total and components)
def plot_dgfit_emission(
    ax,
    hdu,
    obsdata,
    colors=["C3", "C0", "C2", "C1", "C4", "C5", "C6"],
    fontsize=12,
    comps=True,
    ltype="-",
):
    ax.plot(hdu.data["WAVE"], hdu.data["EMIS"], colors[0] + ltype)
    yrange = get_krange(hdu.data["EMIS"], logaxis=True)
    if comps:
        # linetypes = ["-", "-", "-", "-", "-"]
        for i in range(len(hdu.data.names) - 2):

            if np.sum(hdu.data["EMIS" + str(i + 1)]) == 0:
                continue

            ax.plot(
                hdu.data["WAVE"],
                hdu.data["EMIS" + str(i + 1)],
                colors[i + 1] + ltype,
            )
            yrange = get_krange(
                hdu.data["EMIS" + str(i + 1)], logaxis=True, in_range=yrange
            )

    if obsdata.obs_filenames["ir_emis"] is not None:
        if len(obsdata.ir_emission_av) < 25:
            ax.errorbar(
                obsdata.ir_emission_waves,
                obsdata.ir_emission_av,
                yerr=obsdata.ir_emission_av_unc,
                fmt="o",
                label="Observed",
                color="black",
            )
        else:
            ax.plot(
                obsdata.ir_emission_waves,
                obsdata.ir_emission_av,
                "k-",
                label="Observed",
            )
        yrange_obs = get_krange(obsdata.ir_emission_av, logaxis=True)

    ISRF_value = hdu.header["ISRF"]
    ax.set_title(f"ISRF = {ISRF_value:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)
    ax.set_ylabel(r"$S$ $[MJy$ $sr^{-1}$ $A(V)^{-1}]$", fontsize=fontsize)
    ax.set_xlim(get_krange(hdu.data["WAVE"], logaxis=True))
    ax.set_ylim(yrange_obs)


# plot the dust scattering albedo
def plot_dgfit_albedo(
    ax,
    hdu,
    obsdata,
    colors=["C3", "C0", "C2", "C1", "C4", "C5", "C6"],
    fontsize=12,
    comps=True,
    ltype="-",
):
    ax.plot(hdu.data["WAVE"], hdu.data["ALBEDO"], colors[0] + ltype)
    yrange = get_krange(hdu.data["ALBEDO"])
    logscale = True
    if comps:
        # linetypes = ["-", "-", "-", "-", "-"]
        for i in range(len(hdu.data.names) - 2):

            if np.sum(hdu.data["ALBEDO" + str(i + 1)]) == 0:
                continue

            ax.plot(
                hdu.data["WAVE"],
                hdu.data["ALBEDO" + str(i + 1)],
                colors[i + 1] + ltype,
            )
            yrange = get_krange(hdu.data["ALBEDO" + str(i + 1)], in_range=yrange)

    if obsdata.obs_filenames["scat_a"] is not None:
        logscale = False
        ax.errorbar(
            obsdata.scat_a_waves,
            obsdata.scat_albedo,
            yerr=obsdata.scat_albedo_unc,
            fmt="ko",
            label="Observed",
        )

    ax.set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)
    ax.set_ylabel(r"$albedo$", fontsize=fontsize)
    ax.set_xlim(get_krange(hdu.data["WAVE"], logaxis=True))
    ax.set_ylim([0.0, 1.0])
    if logscale:
        ax.set_xscale("log")


# plot the dust scattering phase function asymmetry
def plot_dgfit_g(
    ax,
    hdu,
    obsdata,
    colors=["C3", "C0", "C2", "C1", "C4", "C5", "C6"],
    fontsize=12,
    comps=True,
    ltype="-",
):
    ax.plot(hdu.data["WAVE"], hdu.data["G"], colors[0] + ltype)
    yrange = get_krange(hdu.data["G"])
    logscale = True
    if comps:
        # linetypes = ["-", "-", "-", "-", "-"]
        for i in range(len(hdu.data.names) - 2):

            if np.sum(hdu.data["G" + str(i + 1)]) == 0:
                continue

            ax.plot(
                hdu.data["WAVE"],
                hdu.data["G" + str(i + 1)],
                colors[i + 1] + ltype,
            )
            yrange = get_krange(hdu.data["G" + str(i + 1)], in_range=yrange)

    if obsdata.obs_filenames["scat_g"] is not None:
        logscale = False
        ax.errorbar(
            obsdata.scat_g_waves,
            obsdata.scat_g,
            yerr=obsdata.scat_g_unc,
            fmt="ko",
            label="Observed",
        )

    ax.set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)
    ax.set_ylabel(r"$g$", fontsize=fontsize)
    ax.set_xlim(get_krange(hdu.data["WAVE"], logaxis=True))
    ax.set_ylim([0.0, 1.0])
    if logscale:
        ax.set_xscale("log")


def main():
    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename",
        help=(
            "file with the dust model details "
            + "(size distribution, extinction, etc.)"
        ),
    )
    parser.add_argument(
        "obsfile", help="Data file giving the observational data that was fit"
    )
    parser.add_argument(
        "--start", help="include the starting model", action="store_true"
    )
    parser.add_argument(
        "--markeverynth", type=int, default=2, help="Put a marker every nth point"
    )
    parser.add_argument("--smc", help="use an SMC sightline", action="store_true")
    parser.add_argument(
        "-p", "--png", help="save figure as a png file", action="store_true"
    )
    parser.add_argument(
        "-e", "--eps", help="save figure as an eps file", action="store_true"
    )
    parser.add_argument("-pdf", help="save figure as a pdf file", action="store_true")
    parser.add_argument(
        "--priorfile", default="none", help="Data file giving the prior range"
    )
    args = parser.parse_args()

    # setup the plots
    fontsize = 16
    font = {"size": fontsize}

    matplotlib.rc("font", **font)
    matplotlib.rc("lines", linewidth=2)
    matplotlib.rc("axes", linewidth=2)
    matplotlib.rc("xtick.major", width=2)
    matplotlib.rc("ytick.major", width=2)

    fig, ax = pyplot.subplots(
        ncols=2,
        nrows=4,
        figsize=(15, 16),
        gridspec_kw={"height_ratios": [3, 3, 1, 3]},
    )

    # share x axes between main and residual panels
    ax[2, 0].sharex(ax[1, 0])
    ax[2, 1].sharex(ax[1, 1])

    # open the DGFit results
    hdulist = fits.open(args.filename)

    # get the observed data
    OD = ObsData(args.obsfile)

    # plot the dust size distributions
    plot_dgfit_sizedist(
        ax[0, 0],
        hdulist,
        fontsize=fontsize,
        mass=True,
        plegend=False,
        markers=args.markeverynth,
        file=args.priorfile,
    )

    # plot the abundances
    plot_dgfit_abundances(
        ax[0, 1],
        hdulist,
        OD,
        fontsize=fontsize,
        color="r",
        plegend=True,
        plabel="Model",
    )

    # plot the resulting total and component extinction curves
    plot_dgfit_extinction(ax[1, 0], hdulist["EXTINCTION"], OD, fontsize=fontsize)

    # plot the resulting total and component emission spectra
    plot_dgfit_emission(ax[1, 1], hdulist["EMISSION"], OD, fontsize=fontsize)

    ext_hdu = hdulist["EXTINCTION"]
    residuals_ext = (OD.ext_alav - ext_hdu.data["EXT"]) / OD.ext_alav
    unc_ext = OD.ext_alav_unc / OD.ext_alav
    ax[2, 0].plot(ext_hdu.data["WAVE"], residuals_ext, color="black")
    ax[2, 0].fill_between(
        ext_hdu.data["WAVE"],
        residuals_ext - unc_ext,
        residuals_ext + unc_ext,
        color="k",
        alpha=0.3,
    )
    ax[2, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    ax[2, 0].set_xscale("log")
    ax[2, 0].set_ylim(-0.75, 0.75)
    ax[2, 0].set_ylabel("Residuals\n(model-data)/data", fontsize=fontsize - 4)
    ax[2, 0].set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)

    emis_hdu = hdulist["EMISSION"]
    residuals_emis = (OD.ir_emission_av - emis_hdu.data["EMIS"]) / OD.ir_emission_av
    unc_emis = OD.ir_emission_av_unc / OD.ir_emission_av
    ax[2, 1].plot(emis_hdu.data["WAVE"], residuals_emis, color="black")
    ax[2, 1].fill_between(
        emis_hdu.data["WAVE"],
        residuals_emis - unc_emis,
        residuals_emis + unc_emis,
        color="k",
        alpha=0.3,
    )
    ax[2, 1].axhline(0, color="red", linestyle="--", linewidth=1)
    ax[2, 1].set_xscale("log")
    ax[2, 1].set_ylim(-0.75, 0.75)
    ax[2, 1].set_ylabel("Residuals\n(model-data)/data", fontsize=fontsize - 4)
    ax[2, 1].set_xlabel(r"$\lambda [\mu m]$", fontsize=fontsize)

    # plot the resulting albedos
    plot_dgfit_albedo(ax[3, 0], hdulist["ALBEDO"], OD, fontsize=fontsize)

    # plot the resulting g values
    plot_dgfit_g(ax[3, 1], hdulist["G"], OD, fontsize=fontsize)

    if args.start:
        if "best_fin" in args.filename:
            repstr = "best_fin"
        elif "fin" in args.filename:
            repstr = "fin"
        else:
            repstr = "best_optimizer"
        hdulist2 = fits.open(args.filename.replace(repstr, "start"))
        plot_dgfit_sizedist(
            ax[0, 0],
            hdulist2,
            fontsize=fontsize,
            plegend=False,
            alpha=0.50,
            markers=args.markeverynth,
        )
        plot_dgfit_abundances(ax[0, 1], hdulist2, OD, fontsize=fontsize, color="c")
        plot_dgfit_extinction(
            ax[1, 0], hdulist2["EXTINCTION"], OD, fontsize=fontsize, ltype="--"
        )
        plot_dgfit_emission(
            ax[1, 1], hdulist2["EMISSION"], OD, fontsize=fontsize, ltype="--"
        )
        plot_dgfit_albedo(
            ax[3, 0], hdulist2["ALBEDO"], OD, fontsize=fontsize, ltype="--"
        )
        plot_dgfit_g(ax[3, 1], hdulist2["G"], OD, fontsize=fontsize, ltype="--")

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=fontsize, bbox_to_anchor=(0.5, 1.01))

    pyplot.tight_layout(rect=[0, 0, 1, 0.97])

    # show or save
    basename = args.filename
    basename.replace(".fits", "")
    if args.png:
        fig.savefig(basename + ".png")
    elif args.eps:
        fig.savefig(basename + ".eps")
    elif args.pdf:
        fig.savefig(basename + ".pdf")
    else:
        pyplot.show()


if __name__ == "__main__":
    main()
