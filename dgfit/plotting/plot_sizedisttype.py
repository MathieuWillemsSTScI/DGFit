import argparse
import matplotlib.pyplot as plt
import importlib.resources as importlib_resources
import math

from dgfit.dustmodel import (
    DustModel,
    WD01DustModel,
    ZDA04DustModel,
    Y24DustModel,
    HD23DustModel,
    MRN77DustModel,
)


def set_grains_for_fitting(names):

    grain_list = []
    for grain in names:
        grain_list.append(grain)

    return grain_list


def main():

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--everynth", type=int, default=3, help="Use every nth grain size"
    )
    parser.add_argument(
        "--composition",
        nargs="+",
        default="Carbon",
        choices=[
            "Carbon",
            "Silicates",
        ],
        help="Which grains to show",
    )
    parser.add_argument(
        "--multa4", action="store_true", help="multiply the size distribution by a**4"
    )
    parser.add_argument(
        "--mass", action="store_true", help="Show the mass distribution"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 16
    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(figsize=(14, 9))

    if args.composition == "Carbon":
        composition = [
            "astro-carbonaceous-WD01",
            "PAH-ZDA04",
            "Graphite-ZDA04",
            "ACH2-ZDA04",
            "Carbonaceous-HD23",
            "a-C-Y24",
            "a-C:H-Y24",
        ]

    else:
        composition = [
            "astro-silicates-WD01",
            "Silicates-ZDA04",
            "AstroDust-HD23",
            "aSil-2-Y24",
        ]

    # get the dust model on the full wavelength grid
    compnames = set_grains_for_fitting(composition)
    ref = importlib_resources.files("dgfit") / "data"

    if "astro-silicates-WD01" in compnames or "astro-carbonaceous-WD01" in compnames:
        WD_compnames = [
            name
            for name in compnames
            if "astro-silicates-WD01" in name or "astro-carbonaceous-WD01" in name
        ]
        with importlib_resources.as_file(ref) as data_path:
            dustmodel_WD_full = DustModel(
                componentnames=WD_compnames,
                path=str(data_path) + "/indiv_grain/",
                every_nth=args.everynth,
            )
        dustmodel_WD = WD01DustModel(dustmodel=dustmodel_WD_full)
        # avnhi = 5.3e-22

        # set size distributions
        p0 = []
        for component in dustmodel_WD.components:
            if component.name == "astro-silicates-WD01":
                cparams = dustmodel_WD.parameters["astro-silicates-WD01"]
                p0 += [
                    cparams["C_s"],
                    cparams["a_ts"],
                    cparams["alpha_s"],
                    cparams["beta_s"],
                ]
            else:
                cparams = dustmodel_WD.parameters["astro-carbonaceous-WD01"]
                p0 += [
                    cparams["C_g"],
                    cparams["a_tg"],
                    cparams["alpha_g"],
                    cparams["beta_g"],
                    cparams["a_cg"],
                    cparams["b_C"],
                ]
        cparams = dustmodel_WD.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        dustmodel_WD.set_size_dist(p0)

        # plot size distributions
        markers = ["^", "*"]
        for k, component in enumerate(dustmodel_WD.components):
            if args.multa4:
                sizes = (component.sizes) ** 4
                dist = component.size_dist * sizes
            elif args.mass:
                sizes = (component.sizes) ** 3
                dist = (
                    component.size_dist * sizes * component.density * (4 / 3) * math.pi
                )
            else:
                dist = component.size_dist
            mask = component.size_dist > 0
            dist = dist[mask]
            all_sizes = component.sizes[mask]
            ax.plot(
                all_sizes * 1e4,
                dist,
                label=component.name,
                marker=markers[k],
                markersize=7,
            )

    if (
        "PAH-ZDA04" in compnames
        or "Graphite-ZDA04" in compnames
        or "Silicates-ZDA04" in compnames
        or "ACH2-ZDA04" in compnames
    ):
        Z04_compnames = [
            name
            for name in compnames
            if "PAH-ZDA04" in name
            or "Graphite-ZDA04" in name
            or "Silicates-ZDA04" in name
            or "ACH2-ZDA04" in name
        ]
        with importlib_resources.as_file(ref) as data_path:
            dustmodel_Z04_full = DustModel(
                componentnames=Z04_compnames,
                path=str(data_path) + "/indiv_grain/",
                every_nth=args.everynth,
            )
        dustmodel_Z04 = ZDA04DustModel(dustmodel=dustmodel_Z04_full)
        # avnhi = 5.34e-22

        # set size distributions
        p0 = []
        for component in dustmodel_Z04.components:
            if component.name == "PAH-ZDA04":
                cparams = dustmodel_Z04.parameters["PAH-ZDA04"]
                p0 += [
                    cparams["A"],
                    cparams["c_0"],
                    cparams["b_0"],
                    cparams["b_1"],
                    cparams["m_1"],
                    cparams["a_3"],
                    cparams["m_3"],
                ]

            elif component.name == "Graphite-ZDA04":
                cparams = dustmodel_Z04.parameters["Graphite-ZDA04"]
                p0 += [
                    cparams["A"],
                    cparams["c_0"],
                    cparams["b_0"],
                    cparams["b_1"],
                    cparams["a_1"],
                    cparams["m_1"],
                    cparams["b_3"],
                    cparams["a_3"],
                    cparams["m_3"],
                    cparams["b_4"],
                    cparams["a_4"],
                    cparams["m_4"],
                ]

            elif component.name == "Silicates-ZDA04":
                cparams = dustmodel_Z04.parameters["Silicates-ZDA04"]
                p0 += [
                    cparams["A"],
                    cparams["c_0"],
                    cparams["b_0"],
                    cparams["b_1"],
                    cparams["a_1"],
                    cparams["m_1"],
                    cparams["b_3"],
                    cparams["a_3"],
                    cparams["m_3"],
                    cparams["b_4"],
                    cparams["a_4"],
                    cparams["m_4"],
                ]

            elif component.name == "ACH2-ZDA04":
                cparams = dustmodel_Z04.parameters["ACH2-ZDA04"]
                p0 += [
                    cparams["A"],
                    cparams["c_0"],
                    cparams["b_0"],
                    cparams["b_1"],
                    cparams["a_1"],
                    cparams["m_1"],
                ]
        cparams = dustmodel_Z04.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        dustmodel_Z04.set_size_dist(p0)

        # plot size distributions
        markers = ["d", "h", "x", "v"]
        for k, component in enumerate(dustmodel_Z04.components):
            if args.multa4:
                sizes = (component.sizes) ** 4
                dist = component.size_dist * sizes
            elif args.mass:
                sizes = (component.sizes) ** 3
                dist = (
                    component.size_dist * sizes * component.density * (4 / 3) * math.pi
                )
            else:
                dist = component.size_dist
            mask = component.size_dist > 10
            dist = dist[mask]
            all_sizes = component.sizes[mask]
            ax.plot(
                all_sizes * 1e4,
                dist,
                label=component.name,
                marker=markers[k],
                markersize=7,
            )

    if "Carbonaceous-HD23" in compnames or "AstroDust-HD23" in compnames:
        HD23_compnames = [
            name
            for name in compnames
            if "Carbonaceous-HD23" in name or "AstroDust-HD23" in name
        ]
        with importlib_resources.as_file(ref) as data_path:
            dustmodel_HD23_full = DustModel(
                componentnames=HD23_compnames,
                path=str(data_path) + "/indiv_grain/",
                every_nth=args.everynth,
            )
        dustmodel_HD23 = HD23DustModel(dustmodel=dustmodel_HD23_full)
        # avnhi = 3.2e-22

        p0 = []
        for component in dustmodel_HD23.components:
            if component.name == "Carbonaceous-HD23":
                cparams = dustmodel_HD23.parameters["Carbonaceous-HD23"]
                p0 += [
                    cparams["B_1"],
                    cparams["B_2"],
                ]

            elif component.name == "AstroDust-HD23":
                cparams = dustmodel_HD23.parameters["AstroDust-HD23"]
                p0 += [
                    cparams["B_ad"],
                    cparams["a_0"],
                    cparams["sigma_ad"],
                    cparams["A_0"],
                    cparams["A_1"],
                    cparams["A_2"],
                    cparams["A_3"],
                    cparams["A_4"],
                    cparams["A_5"],
                ]
        cparams = dustmodel_HD23.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        dustmodel_HD23.set_size_dist(p0)

        # plot size distributions
        markers = ["o", "s"]
        for k, component in enumerate(dustmodel_HD23.components):
            if args.multa4:
                sizes = (component.sizes) ** 4
                dist = component.size_dist * sizes
            elif args.mass:
                sizes = (component.sizes) ** 3
                dist = (
                    component.size_dist * sizes * component.density * (4 / 3) * math.pi
                )
            else:
                dist = component.size_dist
            mask = component.size_dist > 0
            dist = dist[mask]
            all_sizes = component.sizes[mask]
            ax.plot(
                all_sizes * 1e4,
                dist,
                label=component.name,
                marker=markers[k],
                markersize=7,
            )

    if "a-C-Y24" in compnames or "a-C:H-Y24" in compnames or "aSil-2-Y24" in compnames:
        Themis_compnames = [
            name
            for name in compnames
            if "a-C-Y24" in name or "a-C:H-Y24" in name or "aSil-2-Y24" in name
        ]

        with importlib_resources.as_file(ref) as data_path:
            dustmodel_Themis_full = DustModel(
                componentnames=Themis_compnames,
                path=str(data_path) + "/indiv_grain/",
                every_nth=args.everynth,
            )
        dustmodel_Themis = Y24DustModel(dustmodel=dustmodel_Themis_full)
        # avnhi = 5.34e-22

        p0 = []
        for component in dustmodel_Themis.components:
            if component.name == "a-C-Y24":
                cparams = dustmodel_Themis.parameters["a-C-Y24"]
                p0 += [
                    cparams["A"],
                    cparams["alpha"],
                    cparams["a_C"],
                    cparams["a_t"],
                    cparams["gamma"],
                ]
            elif component.name == "a-C:H-Y24":
                cparams = dustmodel_Themis.parameters["a-C:H-Y24"]
                p0 += [
                    cparams["A"],
                    cparams["a_0"],
                    cparams["sigma"],
                ]
            elif component.name == "aSil-2-Y24":
                cparams = dustmodel_Themis.parameters["aSil-2-Y24"]
                p0 += [
                    cparams["A"],
                    cparams["a_0"],
                    cparams["sigma"],
                ]

        cparams = dustmodel_Themis.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        dustmodel_Themis.set_size_dist(p0)

        # plot size distributions
        markers = ["p", ">", "<"]
        for k, component in enumerate(dustmodel_Themis.components):
            if args.multa4:
                sizes = (component.sizes) ** 4
                dist = component.size_dist * sizes
            elif args.mass:
                sizes = (component.sizes) ** 3
                dist = (
                    component.size_dist * sizes * component.density * (4 / 3) * math.pi
                )
            else:
                dist = component.size_dist
            mask = component.size_dist > 0
            dist = dist[mask]
            all_sizes = component.sizes[mask]
            ax.plot(
                all_sizes * 1e4,
                dist,
                label=component.name,
                marker=markers[k],
                markersize=7,
            )

    if args.composition[0] == "Carbon":
        density = 2.24
    else:
        density = 3.5
    MRN_compnames = ["astro-silicates-WD01"]
    with importlib_resources.as_file(ref) as data_path:
        dustmodel_MRN_full = DustModel(
            componentnames=MRN_compnames,
            path=str(data_path) + "/indiv_grain/",
            every_nth=args.everynth,
        )
    dustmodel_MRN = MRN77DustModel(dustmodel=dustmodel_MRN_full)
    # avnhi = 3.782e-22

    p0 = []
    for component in dustmodel_MRN.components:
        cparams = dustmodel_MRN.parameters[component.name]
        p0 += [
            cparams["C"],
            cparams["alpha"],
            cparams["a_min"],
            cparams["a_max"],
        ]
    cparams = dustmodel_MRN.parameters["Radiation field"]
    p0 += [cparams["RF"]]
    dustmodel_MRN.set_size_dist(p0)

    for k, component in enumerate(dustmodel_MRN.components):
        size_min = 1e-7
        size_max = 1e-3
        mask = (component.sizes >= size_min) & (component.sizes <= size_max)

        if args.multa4:
            sizes = (component.sizes[mask]) ** 4
            dist = component.size_dist[mask] * sizes
        elif args.mass:
            sizes = (component.sizes[mask]) ** 3
            dist = component.size_dist[mask] * sizes * density * (4 / 3) * math.pi
        else:
            dist = component.size_dist[mask]

        ax.plot(component.sizes[mask] * 1e4, dist, label="MRN77", color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if args.mass:
        ax.set_ylim(1e-3, 1e3)
        ax.set_ylabel(r"$m(a)/A(V)$ [g/A(V)]", fontsize=fontsize)
    elif args.multa4:
        ax.set_ylim(1e-10, 1e-4)
        ax.set_ylabel(r"$a^4f(a)/A(V)$", fontsize=fontsize)
    else:
        ax.set_ylim(1e5, 1e24)
        ax.set_ylabel(r"$dn/da\ A(V)^{-1}$", fontsize=fontsize)
    ax.set_xlim(0.2e-3, 1e1)
    ax.set_xlabel(r"a $[\mu m]$", fontsize=fontsize)
    # ax.set_title("Size distributions")
    ax.legend()

    fig.tight_layout()

    # show or save
    basename = "effsize"
    if args.png:
        fig.savefig(basename + ".png")
    elif args.pdf:
        fig.savefig(basename + ".pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
