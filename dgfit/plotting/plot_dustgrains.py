import argparse
import importlib.resources as importlib_resources

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from dgfit.obsdata import ObsData
from dgfit.dustgrains import DustGrains


def main():

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--composition",
        choices=[
            "astro-silicates",
            "astro-carbonaceous",
            "astro-graphite",
            "PAH-Z04",
            "Graphite-Z04",
            "Silicates-Z04",
            "ACH2-Z04",
            "Silicates1-Z04",
            "Silicates2-Z04",
            "Carbonaceous-HD23",
            "AstroDust-HD23",
            "a-C-Themis",
            "a-C:H-Themis",
            "aSil-2-Themis",
        ],
        default="astro-silicates",
        help="Grain composition",
    )
    parser.add_argument(
        "--obsdata",
        type=str,
        default="none",
        help="transform to observed data grids, with the name of the observed data file as input",
    )
    parser.add_argument(
        "--everynth", type=int, default=5, help="Use every nth grain size"
    )
    parser.add_argument(
        "--ISRF", default=1.0, type=float, help="Choose an ISRF strength"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    DG = DustGrains()
    ref = importlib_resources.files("dgfit") / "data"
    with importlib_resources.as_file(ref) as data_path:
        DG.from_files(
            args.composition,
            path=str(data_path) + "/indiv_grain/",
            every_nth=args.everynth,
        )

    if args.obsdata != "none":
        OD = ObsData(args.obsdata)
        new_DG = DustGrains()
        new_DG.from_object(DG, OD)
        DG = new_DG

    plot(DG, args.composition, args.ISRF, args.png, args.eps, args.pdf)


def plot(DG, composition, ISRF, png=False, eps=False, pdf=False):
    # setup the plots
    fontsize = 12
    font = {"size": fontsize}

    matplotlib.rc("font", **font)
    matplotlib.rc("lines", linewidth=2)
    matplotlib.rc("axes", linewidth=2)
    matplotlib.rc("xtick.major", width=2)
    matplotlib.rc("ytick.major", width=2)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

    ws_indxs = np.argsort(DG.wavelengths)
    ews_indxs = np.argsort(DG.wavelengths_emission)
    waves = DG.wavelengths[ws_indxs]

    num_segments = DG.n_sizes
    DG.sizes *= 10**4
    cmap = get_cmap("jet", num_segments)
    norm = LogNorm(vmin=min(DG.sizes), vmax=max(DG.sizes))
    colors = [cmap(i) for i in range(num_segments)]

    for i in range(DG.n_sizes):
        pcolor = colors[i]

        ax[0, 0].plot(waves, DG.cabs[i, ws_indxs], color=pcolor)
        ax[0, 0].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[0, 0].set_ylabel("C(abs)")
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")

        ax[0, 1].plot(waves, DG.csca[i, ws_indxs], color=pcolor)
        ax[0, 1].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[0, 1].set_ylabel("C(sca)")
        ax[0, 1].set_xscale("log")
        ax[0, 1].set_yscale("log")

        ax[0, 2].plot(waves, DG.cext[i, ws_indxs], color=pcolor)
        ax[0, 2].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[0, 2].set_ylabel("C(ext)")
        ax[0, 2].set_xscale("log")
        ax[0, 2].set_yscale("log")

        ax[1, 0].plot(
            DG.wavelengths_scat_a,
            DG.scat_a_csca[i, :] / DG.scat_a_cext[i, :],
            "o",
            color=pcolor,
        )
        ax[1, 0].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[1, 0].set_ylabel("albedo")
        ax[1, 0].set_xscale("log")
        ax[1, 0].xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=[2.0, 4.0], numticks=10)
        )

        ax[1, 1].plot(DG.wavelengths_scat_g, DG.scat_g[i, :], "o", color=pcolor)
        ax[1, 1].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[1, 1].set_ylabel("g")
        ax[1, 1].set_xscale("log")
        ax[1, 1].xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=[2.0, 4.0], numticks=10)
        )

        emission = DG.interpol_emission(ISRF)

        ax[1, 2].plot(
            DG.wavelengths_emission[ews_indxs],
            emission[i, ews_indxs],
            color=pcolor,
        )
        ax[1, 2].set_xlabel(r"$\lambda$ [$\mu m$]")
        ax[1, 2].set_ylabel("Emission")
        ax[1, 2].set_xscale("log")
        ax[1, 2].set_yscale("log")
        ax[1, 2].set_ylim([1e-23, 1e-0])

    ax[0, 1].set_title(composition)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, aspect=50)
    cbar.set_label(r"Grainsizes [$\mu m$]")
    fig.subplots_adjust(wspace=0.25, right=0.85)

    # show or save
    basename = "DustGrains_diag_%s" % (composition)
    if png:
        fig.savefig(basename + ".png")
    elif eps:
        fig.savefig(basename + ".eps")
    elif pdf:
        fig.savefig(basename + ".pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
