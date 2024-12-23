import argparse
import importlib.resources as importlib_resources

import matplotlib.pyplot as plt
import colorsys
import matplotlib
import numpy as np

from dgfit.obsdata import ObsData
from dgfit.dustgrains import DustGrains


def main():
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wave", default=0.1, type=float, help="lamda in A(lambda)/A(V)"
    )
    parser.add_argument(
        "-c",
        "--composition",
        choices=[
            "astro-silicates",
            "astro-carbonaceous",
            "astro-graphite",
            "astro-PAH-ionized",
            "astro-PAH-neutral",
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
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    DG = DustGrains()
    ref = importlib_resources.files("dgfit") / "data"
    with importlib_resources.as_file(ref) as data_path:
        DG.from_files(args.composition, path=str(data_path) + "/indiv_grain/")

    if args.obsdata != "none":
        OD = ObsData(args.obsdata)
        new_DG = DustGrains()
        new_DG.from_object(DG, OD)
        DG = new_DG

    # setup the plots
    fontsize = 12
    font = {"size": fontsize}

    matplotlib.rc("font", **font)

    matplotlib.rc("lines", linewidth=2)
    matplotlib.rc("axes", linewidth=2)
    matplotlib.rc("xtick.major", width=2)
    matplotlib.rc("ytick.major", width=2)

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(15, 10))

    ws_indxs = np.argsort(DG.wavelengths)
    waves = DG.wavelengths[ws_indxs]
    for i in range(DG.n_sizes):
        pcolor = colorsys.hsv_to_rgb(float(i) / DG.n_sizes / (1.1), 1, 1)

        # get the values at specified lambda and V
        al = np.interp([args.wave, 0.55, 0.45], waves, DG.cext[i, ws_indxs])
        ax[0].plot(DG.sizes[i] * 1e4, al[0] / al[1], "o", color=pcolor)

        rv = al[1] / (al[2] - al[1])
        ax[1].plot(rv, al[0] / al[1], "o", color=pcolor)

    ax[0].set_xlabel(r"$a$ [$\mu m$]")
    ax[0].set_ylabel(f"A({args.wave})/A(V)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")

    ax[1].set_xlabel(r"R(V)")
    ax[1].set_xlim(0.0, 10.0)
    ax[1].set_ylabel(f"A({args.wave})/A(V)")
    ax[1].set_yscale("log")

    ax[0].set_title(args.composition)

    plt.tight_layout()

    # show or save
    basename = "DustGrains_diag_%s" % (args.composition)
    if args.png:
        fig.savefig(basename + ".png")
    elif args.eps:
        fig.savefig(basename + ".eps")
    elif args.pdf:
        fig.savefig(basename + ".pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
