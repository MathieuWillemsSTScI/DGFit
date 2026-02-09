import importlib.resources as importlib_resources
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import h5py

from scipy.optimize import minimize
import emcee
from multiprocessing import Pool
import corner
from nautilus import Prior
from nautilus import Sampler
from scipy.stats import loguniform

from dgfit.dustmodel import (
    DustModel,
    MRN77DustModel,
    WD01DustModel,
    ZDA04DustModel,
    HD23DustModel,
    Y24DustModel,
)
from dgfit.obsdata import ObsData


def DGFit_cmdparser():
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "obsfile", help="Data file giving the observational data to be fit"
    )
    parser.add_argument(
        "--sizedisttype",
        default="WD01",
        choices=["bins", "MRN77", "WD01", "ZDA04", "HD23", "Y24"],
        help="Size distribution type",
    )
    parser.add_argument(
        "--fitobs",
        nargs="+",
        default="all",
        choices=["extinction", "iremission", "abundance", "albedo", "g", "all"],
        help="Which observations to fit",
    )

    parser.add_argument(
        "--composition",
        nargs="+",
        default=["astro-silicates", "astro-carbonaceous"],
        choices=[
            "astro-silicates-WD01",
            "astro-carbonaceous-WD01",
            "PAH-ZDA04",
            "Graphite-ZDA04",
            "Silicates-ZDA04",
            "ACH2-ZDA04",
            "Silicates1-ZDA04",
            "Silicates2-ZDA04",
            "Carbonaceous-HD23",
            "AstroDust-HD23",
            "a-C-Y24",
            "a-C:H-Y24",
            "aSil-2-Y24",
        ],
        help="Which grains to use",
    )

    parser.add_argument(
        "--no_variable_ISRF",
        action="store_false",
        help="disable the variable radiation field",
    )

    parser.add_argument(
        "--mcmc", help="Do MCMC sampling using emcee package", action="store_true"
    )

    parser.add_argument(
        "-f",
        "--fast",
        help="MCMC: Use minimal walkers, steps, burns to debug code",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--slow",
        help="MCMC: Use lots of walkers, n_steps, n_burn",
        action="store_true",
    )
    parser.add_argument(
        "--burnfrac",
        type=float,
        default=0.1,
        help="Fractional portion of nsteps for burn in",
    )
    parser.add_argument(
        "--nsteps", type=int, default=1000, help="Number of samples for full run"
    )
    parser.add_argument(
        "--everynth", type=int, default=2, help="Use every nth grain size"
    )
    parser.add_argument(
        "--chain", action="store_true", help="Store the chain in an ascii file"
    )
    parser.add_argument(
        "--limit_abund", action="store_true", help="Limit based on abundances"
    )
    parser.add_argument(
        "--usemin",
        action="store_true",
        help="Find min before EMCEE (does not work yet)",
    )
    parser.add_argument(
        "-r", "--read", default=None, help="Read size distribution from disk"
    )
    parser.add_argument(
        "-t", "--tag", default="GrainBow_test", help="basename to use for output files"
    )
    parser.add_argument("--nolarge", action="store_true", help="Deweight sizes bigger than the cutoff size")
    parser.add_argument("--cutoff", type=float, default=5.0, help="The cutoff size in micron")
    parser.add_argument(
        "--weight_by_average_unc",
        action="store_true",
        default=False,
        help="weight the observations by the average uncertainty, divide by number of points",
    )
    parser.add_argument(
        "--start_ISRF", type=int, default=1, help="Strength of ISRF to start"
    )
    parser.add_argument(
        "--regularization",
        action="store_true",
        default=False,
        help="add a smoothness criterium to the size distribution",
    )
    parser.add_argument(
        "--fitting_package",
        default="nautilus",
        choices=["nautilus", "emcee"],
        help="Fitting package to use",
    )
    parser.add_argument(
        "--cornerplot",
        action="store_true",
        default=False,
        help="Generate a corner plot of the posterior distributions",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Allows you to run in parallel",
    )
    parser.add_argument(
        "--ncores", type=int, default=4, help="Number of cores to use if you run parallel"
    )
    parser.add_argument(
        "--nlivepoints", type=int, default=2000, help="Number of live points to use for nautilus"
    )
    parser.add_argument(
        "--result_from_file", default="none", help="give the name of the file with the parameters"
    )

    return parser


def set_obs_for_fitting(obsdata, fitobs):
    """
    parse the requested list of observations for fitting and set the
    appropriate variables
    """

    fitobs_list = []
    if not (obsdata.fit_extinction and (("extinction" in fitobs) or ("all" in fitobs))):
        obsdata.fit_extinction = False
    else:
        fitobs_list.append("extinction")
    if not (obsdata.fit_abundance and (("abundance" in fitobs) or ("all" in fitobs))):
        obsdata.fit_abundance = False
    else:
        fitobs_list.append("abundance")
    if not (
        obsdata.fit_ir_emission and (("iremission" in fitobs) or ("all" in fitobs))
    ):
        obsdata.fit_ir_emission = False
    else:
        fitobs_list.append("ir_emission")
    if not (obsdata.fit_scat_a and (("albedo" in fitobs) or ("all" in fitobs))):
        obsdata.fit_scat_a = False
    else:
        fitobs_list.append("scat albedo")
    if not (obsdata.fit_scat_g and (("g" in fitobs) or ("all" in fitobs))):
        obsdata.fit_scat_g = False
    else:
        fitobs_list.append("scat g")

    return fitobs_list


def set_grains_for_fitting(names):

    grain_list = []
    for grain in names:
        grain_list.append(grain)

    return grain_list


def calc_sizedist_fact(dustmodel, obsdata):

    results = dustmodel.eff_grain_props(obsdata)
    natoms = results["natoms"]
    factor_C = 1
    factor_sil = 1
    for atomname in natoms.keys():
        if natoms[atomname] > (
            obsdata.abundance_av[atomname][0] + obsdata.abundance_av[atomname][1]
        ):
            if atomname == "C":
                newfactor_C = natoms[atomname] / obsdata.abundance_av[atomname][0]
                if newfactor_C > factor_C:
                    factor_C = newfactor_C
            else:
                newfactor_sil = natoms[atomname] / obsdata.abundance_av[atomname][0]
                if newfactor_sil > factor_sil:
                    factor_sil = newfactor_sil

    return factor_C, factor_sil


def setparams_MRN77(dustmodel, obsdata, factor_C, factor_sil, ISRF):

    pnames = []
    p0 = []
    deltas = []
    logs = []
    for component in dustmodel.components:
        cparams = dustmodel.parameters[component.name]
        cdeltas = dustmodel.deltas[component.name]
        clogs = dustmodel.logs[component.name]
        if component.name in [
            "astro-silicates-WD01",
            "Silicates-ZDA04",
            "Silicates1-ZDA04",
            "Silicates2-ZDA04",
            "AstroDust-HD23",
            "aSil-2-Y24",
        ]:
            p0 += [
                cparams["C"] / (factor_sil),
            ]
            deltas += [
                np.array(cdeltas["C"]) / (factor_sil),
            ]
        else:
            p0 += [
                cparams["C"] / (factor_C),
            ]
            deltas += [
                np.array(cdeltas["C"]) / (factor_C),
            ]
        p0 += [
            cparams["alpha"],
            cparams["a_min"],
            cparams["a_max"],
        ]
        deltas += [
            cdeltas["alpha"],
            cdeltas["a_min"],
            cdeltas["a_max"],
        ]
        logs += [
            clogs["C"],
            clogs["alpha"],
            clogs["a_min"],
            clogs["a_max"],
        ]
        pnames += cparams.keys()

    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        pnames += cparams.keys()

    return p0, deltas, logs, pnames


def setparams_WD01(dustmodel, obsdata, factor_C, factor_sil, ISRF):

    pnames = []
    p0 = []
    deltas = []
    logs = []
    for component in dustmodel.components:
        if component.name == "astro-silicates-WD01":
            cparams = dustmodel.parameters["astro-silicates-WD01"]
            cdeltas = dustmodel.deltas["astro-silicates-WD01"]
            clogs = dustmodel.logs["astro-silicates-WD01"]
            p0 += [
                cparams["C_s"] / (factor_sil),
                cparams["a_ts"],
                cparams["alpha_s"],
                cparams["beta_s"],
            ]
            deltas += [
                np.array(cdeltas["C_s"]) / (factor_sil),
                cdeltas["a_ts"],
                cdeltas["alpha_s"],
                cdeltas["beta_s"],
            ]
            logs += [
                clogs["C_s"],
                clogs["a_ts"],
                clogs["alpha_s"],
                clogs["beta_s"],
            ]
        else:
            cparams = dustmodel.parameters["astro-carbonaceous-WD01"]
            cdeltas = dustmodel.deltas["astro-carbonaceous-WD01"]
            clogs = dustmodel.logs["astro-carbonaceous-WD01"]
            p0 += [
                cparams["C_g"] / (factor_C),
                cparams["a_tg"],
                cparams["alpha_g"],
                cparams["beta_g"],
                cparams["a_cg"],
                cparams["b_C"],
            ]
            deltas += [
                np.array(cdeltas["C_g"]) / (factor_C),
                cdeltas["a_tg"],
                cdeltas["alpha_g"],
                cdeltas["beta_g"],
                cdeltas["a_cg"],
                cdeltas["b_C"],
            ]
            logs += [
                clogs["C_g"],
                clogs["a_tg"],
                clogs["alpha_g"],
                clogs["beta_g"],
                clogs["a_cg"],
                clogs["b_C"],
            ]
        pnames += cparams.keys()

    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        pnames += cparams.keys()

    return p0, deltas, logs, pnames


def setparams_ZDA04(dustmodel, obsdata, factor_C, factor_sil, ISRF):

    pnames = []
    p0 = []
    deltas = []
    logs = []
    for component in dustmodel.components:
        if component.name == "PAH-ZDA04":
            cparams = dustmodel.parameters["PAH-ZDA04"]
            cdeltas = dustmodel.deltas["PAH-ZDA04"]
            clogs = dustmodel.logs["PAH-ZDA04"]
            p0 += [
                cparams["A"] / (factor_C),
                cparams["c_0"],
                cparams["b_0"],
                cparams["b_1"],
                cparams["m_1"],
                cparams["a_3"],
                cparams["m_3"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_C),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["m_1"],
                cdeltas["a_3"],
                cdeltas["m_3"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["m_1"],
                clogs["a_3"],
                clogs["m_3"],
            ]

        elif component.name == "Graphite-ZDA04":
            cparams = dustmodel.parameters["Graphite-ZDA04"]
            cdeltas = dustmodel.deltas["Graphite-ZDA04"]
            clogs = dustmodel.logs["Graphite-ZDA04"]
            p0 += [
                cparams["A"] / (factor_C),
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
            deltas += [
                np.array(cdeltas["A"]) / (factor_C),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["a_1"],
                cdeltas["m_1"],
                cdeltas["b_3"],
                cdeltas["a_3"],
                cdeltas["m_3"],
                cdeltas["b_4"],
                cdeltas["a_4"],
                cdeltas["m_4"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["a_1"],
                clogs["m_1"],
                clogs["b_3"],
                clogs["a_3"],
                clogs["m_3"],
                clogs["b_4"],
                clogs["a_4"],
                clogs["m_4"],
            ]

        elif component.name == "Silicates-ZDA04":
            cparams = dustmodel.parameters["Silicates-ZDA04"]
            cdeltas = dustmodel.deltas["Silicates-ZDA04"]
            clogs = dustmodel.logs["Silicates-ZDA04"]
            p0 += [
                cparams["A"] / (factor_sil),
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
            deltas += [
                np.array(cdeltas["A"]) / (factor_sil),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["a_1"],
                cdeltas["m_1"],
                cdeltas["b_3"],
                cdeltas["a_3"],
                cdeltas["m_3"],
                cdeltas["b_4"],
                cdeltas["a_4"],
                cdeltas["m_4"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["a_1"],
                clogs["m_1"],
                clogs["b_3"],
                clogs["a_3"],
                clogs["m_3"],
                clogs["b_4"],
                clogs["a_4"],
                clogs["m_4"],
            ]

        elif component.name == "ACH2-ZDA04":
            cparams = dustmodel.parameters["ACH2-ZDA04"]
            cdeltas = dustmodel.deltas["ACH2-ZDA04"]
            clogs = dustmodel.logs["ACH2-ZDA04"]
            p0 += [
                cparams["A"] / (factor_C),
                cparams["c_0"],
                cparams["b_0"],
                cparams["b_1"],
                cparams["a_1"],
                cparams["m_1"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_C),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["a_1"],
                cdeltas["m_1"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["a_1"],
                clogs["m_1"],
            ]

        elif component.name == "Silicates1-ZDA04":
            cparams = dustmodel.parameters["Silicates1-ZDA04"]
            cdeltas = dustmodel.deltas["Silicates1-ZDA04"]
            clogs = dustmodel.logs["Silicates1-ZDA04"]
            p0 += [
                cparams["A"] / (factor_sil),
                cparams["c_0"],
                cparams["b_0"],
                cparams["b_1"],
                cparams["a_1"],
                cparams["m_1"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_sil),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["a_1"],
                cdeltas["m_1"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["a_1"],
                clogs["m_1"],
            ]

        elif component.name == "Silicates2-ZDA04":
            cparams = dustmodel.parameters["Silicates2-ZDA04"]
            cdeltas = dustmodel.deltas["Silicates2-ZDA04"]
            clogs = dustmodel.logs["Silicates2-ZDA04"]
            p0 += [
                cparams["A"] / (factor_sil),
                cparams["c_0"],
                cparams["b_0"],
                cparams["b_1"],
                cparams["a_1"],
                cparams["m_1"],
                cparams["b_2"],
                cparams["a_2"],
                cparams["m_2"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_sil),
                cdeltas["c_0"],
                cdeltas["b_0"],
                cdeltas["b_1"],
                cdeltas["a_1"],
                cdeltas["m_1"],
                cdeltas["b_2"],
                cdeltas["a_2"],
                cdeltas["m_2"],
            ]
            logs += [
                clogs["A"],
                clogs["c_0"],
                clogs["b_0"],
                clogs["b_1"],
                clogs["a_1"],
                clogs["m_1"],
                clogs["b_2"],
                clogs["a_2"],
                clogs["m_2"],
            ]

        pnames += cparams.keys()

    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        pnames += cparams.keys()

    return p0, deltas, logs, pnames


def setparams_HD23(dustmodel, obsdata, factor_C, factor_sil, ISRF):

    pnames = []
    p0 = []
    deltas = []
    logs = []
    for component in dustmodel.components:
        if component.name == "Carbonaceous-HD23":
            cparams = dustmodel.parameters["Carbonaceous-HD23"]
            cdeltas = dustmodel.deltas["Carbonaceous-HD23"]
            clogs = dustmodel.logs["Carbonaceous-HD23"]
            p0 += [
                cparams["B_1"] / (factor_C),
                cparams["B_2"] / (factor_C),
            ]
            deltas += [
                np.array(cdeltas["B_1"]) / (factor_C),
                np.array(cdeltas["B_2"]) / (factor_C),
            ]
            logs += [
                clogs["B_1"],
                clogs["B_2"],
            ]
        elif component.name == "AstroDust-HD23":
            cparams = dustmodel.parameters["AstroDust-HD23"]
            cdeltas = dustmodel.deltas["AstroDust-HD23"]
            clogs = dustmodel.logs["AstroDust-HD23"]
            p0 += [
                cparams["B_ad"] / (factor_sil),
                cparams["a_0"],
                cparams["sigma_ad"],
                cparams["A_0"] / (factor_sil),
                cparams["A_1"],
                cparams["A_2"],
                cparams["A_3"],
                cparams["A_4"],
                cparams["A_5"],
            ]
            deltas += [
                np.array(cdeltas["B_ad"]) / (factor_sil),
                cdeltas["a_0"],
                cdeltas["sigma_ad"],
                np.array(cdeltas["A_0"]) / (factor_sil),
                cdeltas["A_1"],
                cdeltas["A_2"],
                cdeltas["A_3"],
                cdeltas["A_4"],
                cdeltas["A_5"],
            ]
            logs += [
                clogs["B_ad"],
                clogs["a_0"],
                clogs["sigma_ad"],
                clogs["A_0"],
                clogs["A_1"],
                clogs["A_2"],
                clogs["A_3"],
                clogs["A_4"],
                clogs["A_5"],
            ]
        pnames += cparams.keys()

    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        pnames += cparams.keys()

    return p0, deltas, logs, pnames


def setparams_Y24(dustmodel, obsdata, factor_C, factor_sil, ISRF):

    pnames = []
    p0 = []
    deltas = []
    logs = []
    for component in dustmodel.components:
        if component.name == "a-C-Y24":
            cparams = dustmodel.parameters["a-C-Y24"]
            cdeltas = dustmodel.deltas["a-C-Y24"]
            clogs = dustmodel.logs["a-C-Y24"]
            p0 += [
                cparams["A"] / (factor_C),
                cparams["alpha"],
                cparams["a_C"],
                cparams["a_t"],
                cparams["gamma"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_C),
                cdeltas["alpha"],
                cdeltas["a_C"],
                cdeltas["a_t"],
                cdeltas["gamma"],
            ]
            logs += [
                clogs["A"],
                clogs["alpha"],
                clogs["a_C"],
                clogs["a_t"],
                clogs["gamma"],
            ]
        elif component.name == "a-C:H-Y24":
            cparams = dustmodel.parameters["a-C:H-Y24"]
            cdeltas = dustmodel.deltas["a-C:H-Y24"]
            clogs = dustmodel.logs["a-C:H-Y24"]
            p0 += [
                cparams["A"] / (factor_C),
                cparams["a_0"],
                cparams["sigma"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_C),
                cdeltas["a_0"],
                cdeltas["sigma"],
            ]
            logs += [
                clogs["A"],
                clogs["a_0"],
                clogs["sigma"],
            ]
        elif component.name == "aSil-2-Y24":
            cparams = dustmodel.parameters["aSil-2-Y24"]
            cdeltas = dustmodel.deltas["aSil-2-Y24"]
            clogs = dustmodel.logs["aSil-2-Y24"]
            p0 += [
                cparams["A"] / (factor_sil),
                cparams["a_0"],
                cparams["sigma"],
            ]
            deltas += [
                np.array(cdeltas["A"]) / (factor_sil),
                cdeltas["a_0"],
                cdeltas["sigma"],
            ]
            logs += [
                clogs["A"],
                clogs["a_0"],
                clogs["sigma"],
            ]
        pnames += cparams.keys()

    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0 += [cparams["RF"]]
        pnames += cparams.keys()

    return p0, deltas, logs, pnames


def add_priors_nautilus(pnames, logs, deltas, prior):
    for i, name in enumerate(pnames):
                if name != "RF":
                    if logs[i]:
                        prior.add_parameter(
                            f"{name}", dist=loguniform(deltas[i][0], deltas[i][1])
                        )
                    else:
                        prior.add_parameter(
                            f"{name}", dist=(deltas[i][0], deltas[i][1])
                        )

                else:
                    prior.add_parameter("RF", dist=(0.25, 20))
                    logs.append(False)


def main():
    parser = DGFit_cmdparser()

    args = parser.parse_args()

    # set the basename of the output
    basename = f"{args.tag}_{args.sizedisttype}"

    # save the start time
    start_time = time.process_time()

    if args.fitting_package == "emcee":
        # emcee parameters
        if args.fast:
            print("using the fast params")
            nsteps = 100
            burnfrac = 0.1
        elif args.slow:
            print("using the slow params")
            nsteps = 10000
            burnfrac = 0.2
        else:
            burnfrac = float(args.burnfrac)
            nsteps = int(args.nsteps)

    # get the location of the provided data
    ref = importlib_resources.files("dgfit") / "data"

    # get the observed data
    obsdata = ObsData(args.obsfile)

    # determine what to fit based on what exists and the commandline args
    fitobs_list = set_obs_for_fitting(obsdata, args.fitobs)

    # get the dust model on the full wavelength grid
    compnames = set_grains_for_fitting(args.composition)
    with importlib_resources.as_file(ref) as data_path:
        dustmodel_full = DustModel(
            componentnames=compnames,
            path=str(data_path) + "/indiv_grain/",
            every_nth=args.everynth,
            limit_abundances=args.limit_abund,
            variable_ISRF=args.no_variable_ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            regularization=args.regularization,
        )
    for i, comp in enumerate(compnames):
        print(
            f"# of grain sizes for {comp} = {len(dustmodel_full.components[i].sizes)}"
        )

    prior = Prior()
    sizedisttype = args.sizedisttype
    ISRF = args.no_variable_ISRF
    pnames = []
    if sizedisttype == "MRN77":
        # define the fitting model
        dustmodel = MRN77DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
        )

        p0, deltas, logs, _pnames = setparams_MRN77(dustmodel, obsdata, 1, 1, ISRF)
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, deltas, logs, _pnames_ = setparams_MRN77(
                dustmodel, obsdata, factor_C, factor_sil, ISRF
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, deltas, prior)

    elif sizedisttype == "WD01":
        dustmodel = WD01DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
        )

        p0, deltas, logs, _pnames = setparams_WD01(dustmodel, obsdata, 1, 1, ISRF)
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, deltas, logs, _pnames_ = setparams_WD01(
                dustmodel, obsdata, factor_C, factor_sil, ISRF
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, deltas, prior)

    elif sizedisttype == "ZDA04":
        dustmodel = ZDA04DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
        )

        p0, deltas, logs, _pnames = setparams_ZDA04(dustmodel, obsdata, 1, 1, ISRF)
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, deltas, logs, _pnames_ = setparams_ZDA04(
                dustmodel, obsdata, factor_C, factor_sil, ISRF
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, deltas, prior)

    elif sizedisttype == "HD23":
        dustmodel = HD23DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
        )

        p0, deltas, logs, _pnames = setparams_HD23(dustmodel, obsdata, 1, 1, ISRF)
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, deltas, logs, _pnames_ = setparams_HD23(
                dustmodel, obsdata, factor_C, factor_sil, ISRF
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, deltas, prior)

    elif sizedisttype == "Y24":
        dustmodel = Y24DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
        )

        p0, deltas, logs, _pnames = setparams_Y24(dustmodel, obsdata, 1, 1, ISRF)
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, deltas, logs, _pnames_ = setparams_Y24(
                dustmodel, obsdata, factor_C, factor_sil, ISRF
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, deltas, prior)

    # replace the default size distribution with one from a file
    elif args.read is not None:
        dustmodel.read_sizedist_from_file(args.read)

    elif sizedisttype == "bins":
        if args.fitting_package != "nautilus":
            print("Only the nautilus sampling package is supported for bins")
            exit()
        dustmodel = DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            regularization=args.regularization,
        )

        eps = 1e-6
        p0 = []
        lowers = []
        uppers = []
        used_sizes = []
        for k in range(0, dustmodel.n_components):
            n_grains = dustmodel.calculate_priors(dustmodel.components[k], obsdata)
            for kk in range(len(dustmodel.components[k].size_dist)):
                if args.nolarge:
                    if dustmodel.components[k].sizes[kk] > (args.cutoff * 1e-4):
                        print("Deweighting size", dustmodel.components[k].sizes[kk]*10000, "microns")
                        prior.add_parameter(f"c{k + 1}_s{kk}", dist=0)
                        p0.append(0)
                        lowers.append(0)
                        uppers.append(0)
                        continue
                pnames += [f"c{k + 1}_s{kk}"]
                upper = n_grains[kk]
                lower = (dustmodel.components[k].size_dist[kk] / 1e7) / 100000
                upper *= 1 + (eps * (k + kk + 1))
                lower *= 1 - (eps * (k + kk + 1))
                prior.add_parameter(f"c{k + 1}_s{kk + 1}", dist=loguniform(lower, upper))
                p0.append(dustmodel.components[k].size_dist[kk] / 1e7)
                lowers.append(lower)
                uppers.append(upper)
                used_sizes.append(dustmodel.components[k].sizes[kk])

        if ISRF:
            pnames += ["RF"]
            prior.add_parameter("RF", dist=(0.25, 20))
            p0.append(1)
            lowers.append(0.25)
            uppers.append(20)

        dustmodel.set_size_dist(p0)
        np.savetxt(f"priors_{args.tag}.txt", np.column_stack((uppers, lowers)), fmt="%.10e")

    else:
        print("Size distribution choice not known")
        exit()

    # save the starting model
    dustmodel.save_results(basename + "_sizedist_start.fits", obsdata)

    # setup time
    setup_time = time.process_time()
    print("setup time taken: ", (setup_time - start_time) / 60.0, " min")

    if args.fitting_package == "nautilus":

        def loglike(a):
            x = np.array(list(a.values()))
            return dustmodel.lnprob(x, obsdata, dustmodel)
        
        if args.result_from_file == "none":

            if args.parallel:
                sampler = Sampler(
                    prior, loglike, n_live=args.nlivepoints, filepath=f"checkpoint_{basename}_sizedist.hdf5", pool=args.ncores
                )
            else:
                sampler = Sampler(
                    prior, loglike, n_live=args.nlivepoints, filepath=f"checkpoint_{basename}_sizedist.hdf5"
                )
            sampler.run(verbose=True)
            opt_time = time.process_time()
            print("optimizer time taken: ", (opt_time - setup_time) / 60.0, " min")
            print(f"Evidence: {sampler.log_z}")

            points, log_w, log_l = sampler.posterior()
            map_index = np.argmax(log_l)
            opt_params = points[map_index]
            weights = np.exp(log_w)
            labels = np.array(prior.keys)

            with h5py.File(f"posterior_samples_{args.tag}.h5", "w") as f:
                f["points"] = points
                f["log_w"]  = log_w
                f["log_l"]  = log_l
                f["labels"] = labels.astype("S")

            if args.cornerplot:
                n_params = points.shape[1]
                chunk_size = 5

                for start in range(0, n_params, chunk_size):
                    end = start + chunk_size
                    pts_subset = points[:, start:end]
                    labels_subset = labels[start:end]
                    opt_subset = opt_params[start:end]

                    fig = corner.corner(
                        pts_subset,
                        weights=weights,
                        bins=100,
                        labels=labels_subset,
                        color="purple",
                        show_titles=True,
                        title_fmt=".3g",
                        quantiles=[0.16, 0.5, 0.84],
                        levels=[0.68, 0.95],
                        plot_datapoints=False,
                        range=np.repeat(0.999, len(labels_subset)),
                    )
                    corner.overplot_lines(fig, opt_subset, color="red")
                    plt.show()
                    plt.close(fig)
            
        else:
            x_max = np.loadtxt(f"{args.result_from_file}")
            opt = np.array(x_max)
            lowers = np.array(lowers)
            uppers = np.array(uppers)
            opt_params = np.zeros(len(uppers))
            p = 0
            for i, value in enumerate(uppers):
                if value != 0:
                    k = i - p
                    opt_params[i] = (lowers[i] * (uppers[i]/lowers[i])**opt[k])
                else:
                    opt_params[i] = 0
                    p += 1
            if ISRF:
                opt_params[-1] = lowers[-1] + (opt[-1] * (uppers[-1] - lowers[-1]))

        dustmodel.set_size_dist(opt_params)
        print(f"ln(p): {dustmodel.lnprob(opt_params, obsdata, dustmodel)}")
        oname = f"{basename}_sizedist_best_optimizer.fits"
        dustmodel.save_results(oname, obsdata)

    if args.fitting_package == "emcee":
        # do simple optimization to find the best fit
        call_count = {"n": 0}

        def nll(*args):
            call_count["n"] += 1
            if call_count["n"] % 1000 == 0:
                print(
                    f"Call {call_count['n']}: ln(p) = {-dustmodel.lnprob(*args)}"
                )  # added this line to check when the minimizer converges
                print(f"Number of points: {dustmodel.fracs[5]}")
                print(
                    f"Extinction: {round(np.abs(dustmodel.fracs[0]) * 100, 2)}% ({obsdata.ext_npts})"
                )
                print(
                    f"Emission: {round(np.abs(dustmodel.fracs[2]) * 100, 2)}% ({obsdata.ir_emission_npts})"
                )
                print(
                    f"Abundance: {round(np.abs(dustmodel.fracs[1]) * 100, 2)}% ({obsdata.abundance_npts})"
                )
                print(
                    f"Albedo: {round(np.abs(dustmodel.fracs[3]) * 100, 2)}% ({obsdata.scat_a_npts})"
                )
                print(
                    f"g: {round(np.abs(dustmodel.fracs[4]) * 100, 2)}% ({obsdata.scat_g_npts})"
                )
                comps = dustmodel.components
                grain = comps[0]
                print(f"ISRF: {grain.RF_strength}")
            return -dustmodel.lnprob(*args)

        soln = minimize(
            nll,
            p0,
            args=(obsdata, dustmodel),
            method="Nelder-Mead",
            options={"maxiter": 500000, "maxfev": 500000, "disp": True},
        )
        opt_params = soln.x
        dustmodel.set_size_dist_parameters(opt_params)
        opt_time = time.process_time()
        print("optimizer time taken: ", (opt_time - setup_time) / 60.0, " min")

        if args.mcmc:
            p0 = opt_params
            # more emcee setup
            ndim = len(p0)
            nwalkers = 2 * ndim

            print(f"fitting {fitobs_list}")
            print(f"# params = {ndim}")
            print(f"# walkers = {nwalkers}")
            print(f"# burnfrac = {burnfrac}")
            print(f"# steps = {nsteps}")

            # setting up the walkers to start "near" the inital guess
            p = dustmodel.initial_walkers(p0, nwalkers)

            # Set up the backend to save the samples for the emcee runs
            emcee_samples_file = f"{basename}_chain.h5"
            backend = emcee.backends.HDFBackend(emcee_samples_file)
            backend.reset(nwalkers, ndim)

            # setup the sampler
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    dustmodel.lnprob,
                    args=(obsdata, dustmodel),
                    pool=pool,
                    backend=backend,
                )

                # do the sampling
                sampler.run_mcmc(p, nsteps, progress=True)

            emcee_time = time.process_time()
            print("emcee time taken: ", (emcee_time - opt_time) / 60.0, " min")

            # best fit dust params
            oname = "%s_sizedist_best_fin.fits" % (basename)
            dustmodel.save_best_results(oname, sampler, obsdata)

            # 50p dust params
            oname = "%s_sizedist_fin.fits" % (basename)
            dustmodel.save_50percentile_results(
                oname, sampler, obsdata, nburn=int(burnfrac * nsteps)
            )

            if args.cornerplot and ndim < 30:
                # plot the walker chains for all parameters
                nwalkers, nsteps, ndim = sampler.chain.shape
                fig, ax = plt.subplots(ndim, sharex=True, figsize=(13, 13))
                walk_val = np.arange(nsteps)
                for i in range(ndim):
                    for k in range(nwalkers):
                        ax[i].plot(walk_val, sampler.chain[k, :, i], "-")
                        ax[i].set_ylabel(pnames[i])
                fig.savefig(f"{basename}_walker_param_values.png")
                plt.close(fig)

                # plot the 1D and 2D likelihood functions in a traditional triangle plot
                nwalkers, nsteps = sampler.lnprobability.shape
                # discard the 1st burn_frac (burn in)
                flat_samples = sampler.get_chain(
                    discard=int(burnfrac * nsteps), flat=True
                )
                nflatsteps, ndim = flat_samples.shape
                fig = corner.corner(
                    flat_samples,
                    labels=pnames,
                    show_titles=True,
                    title_fmt=".3f",
                    use_math_text=True,
                )
                plt.show()
                fig.savefig(f"{basename}_param_triangle.png")


if __name__ == "__main__":
    main()
