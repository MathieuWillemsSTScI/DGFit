import glob
import re
import math

import numpy as np
from astropy.table import Table

from scipy.interpolate import interp1d

__all__ = ["DustGrains"]


# Object for the proprerties of dust grain with a specific composition
class DustGrains(object):
    """
    DustGrains Class

    dust grain properties stored by dust size/composition

    Attributes
    ----------
    origin : 'string'

    """

    def __init__(self):
        """
        Simple initialization allowing for multiple origins of data
        """
        self.origin = None

    def from_files(self, componentname, path="./", every_nth=5):
        """
        Read in precomputed dust grain information from files.

        Parameters
        ----------
        componentname : 'string'
            Name that givesn the dust composition
            [astro-silicates, astro-carbonacenous, astro-graphite]
        path : 'string'
            Path to the location of the dust grain files
        every_nth : int
            Only use every nth size, faster fitting
        """

        self.origin = "files"

        # min/max wavelengths for storage
        #    set here in case later we want to pass them via the function call
        min_wave = 0.0
        max_wave = (1e6,)
        min_wave_emission = 0.0
        max_wave_emission = 1e6

        # check that the component name is allowed
        _allowed_components = [
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
        ]
        if componentname not in _allowed_components:
            print(componentname + " not one of the allowed grain components")
            print(_allowed_components)
            exit()

        # set useful quantities for each composition
        if componentname in [
            "astro-silicates",
            "Silicates-Z04",
            "Silicates1-Z04",
            "Silicates2-Z04",
        ]:  # from WD01
            self.density = 3.5  # g/cm^3
            self.atomic_composition = "MgFeSiO4"
            self.atomic_comp_names = ["Mg", "Fe", "Si", "O"]
            self.atomic_comp_number = np.array([1, 1, 1, 4])
            self.atomic_comp_masses = (
                np.array([24.305, 55.845, 28.0855, 15.994]) * 1.660e-24
            )  # in grams

        elif componentname in ["aSil-2-Themis"]:  # from Demyk et al. 2022
            self.density = 2.7  # g/cm^3
            self.atomic_composition = "MgSiO4"
            self.atomic_comp_names = ["Mg", "Si", "O", "C"]
            self.atomic_comp_number = np.array([1.7, 1, 3.7, 1])
            self.atomic_comp_masses = (
                np.array([24.305, 28.0855, 15.994, 12.0107]) * 1.660e-24
            )  # in grams

        elif componentname in ["a-C-Themis"]:  # from Themis (2017)
            self.density = 1.6  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["a-C:H-Themis"]:  # from Themis (2017)
            self.density = 1.3  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["astro-carbonaceous", "PAH-Z04"]:  # from WD01
            self.density = 2.24  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["ACH2-Z04"]:  # from Zubko 1996
            self.density = 1.81  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["astro-graphite", "Graphite-Z04"]:  # need origin (copy)
            self.density = 2.24  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["Carbonaceous-HD23"]:  # Draine et al 2021
            self.density = 2.2  # g/cm^3
            self.atomic_composition = "C"
            self.atomic_comp_names = ["C"]
            self.atomic_comp_number = np.array([1])
            self.atomic_comp_masses = np.array([12.0107]) * 1.660e-24  # in grams

        elif componentname in ["AstroDust-HD23"]:
            self.density = 2.74  # g/cm^3
            self.atomic_composition = "MgFeSiO"
            self.atomic_comp_names = ["Mg", "Fe", "Si", "O"]
            self.atomic_comp_number = np.array([1.3, 0.3, 1, 3.6])
            self.atomic_comp_masses = (
                np.array([24.305, 55.845, 28.0855, 15.994]) * 1.660e-24
            )  # in grams

        # useful quantities
        self.mass_per_mol_comp = np.sum(
            self.atomic_comp_masses * self.atomic_comp_number
        )
        self.col_den_constant = (
            (4.0 / 3.0)
            * math.pi
            * self.density
            * (self.atomic_comp_number / self.mass_per_mol_comp)
        )

        # get the filenames of this component for all sizes
        filelist = []
        for file in glob.glob(
            path + "INDIV-GRAINS-DGFIT_c_*" + componentname + "*.dat"
        ):
            m = re.search("_s_(.+?).dat", file)
            if m:
                found = m.group(1)
                sizenum = found

                # get the grain size
                f = open(file, "r")
                firstline = f.readline()
                space_pos = firstline.find(" ", 5)
                f.close()

                filelist.append(
                    (
                        file,
                        int(sizenum),
                        float(firstline[1:space_pos]),
                    )
                )

        # check if any files were found
        if len(filelist) == 0:
            print("no files found")
            print("path = " + path)
            exit()

        # code to just pick every nth grain size
        # makes the fitting faster, but the size distributions coarser
        tindxs = np.arange(0, len(filelist), every_nth)
        sfilelist = sorted(filelist, key=lambda file: file[1])
        filelist = []
        for k in tindxs:
            filelist.append(sfilelist[k])

        # setup the variables to store the grain information
        self.name = componentname
        self.n_sizes = len(filelist)
        self.sizes = np.empty(self.n_sizes)
        self.size_dist = np.empty(self.n_sizes)
        self.stochastic_heating = np.empty(self.n_sizes)
        self.RF_strength = 1

        # loop over the files from the smallest to the largest sizes
        for k, file in enumerate(sorted(filelist, key=lambda file: file[1])):
            # read in the table of grain properties for this size
            t = Table.read(file[0], format="ascii.commented_header", header_start=-1)

            stochheated = False
            for tcomment in t.meta["comments"]:
                if "StochasticallyHeated" in tcomment:
                    if "1" in tcomment:
                        stochheated = True

            # setup more variables now that we know the number of wavelengths
            if k == 0:
                # generate the indices to crop the wavelength to the
                #      desired range
                (gindxs,) = np.where(
                    (t["Wavelength"] >= min_wave) & (t["Wavelength"] <= max_wave)
                )
                (egindxs,) = np.where(
                    (t["Wavelength"] >= min_wave_emission)
                    & (t["Wavelength"] <= max_wave_emission)
                )
                self.wavelengths = np.array(t["Wavelength"][gindxs])
                self.wavelengths_emission = np.array(t["Wavelength"][egindxs])
                self.n_wavelengths = len(self.wavelengths)
                self.n_wavelengths_emission = len(self.wavelengths_emission)
                self.cext = np.empty((self.n_sizes, self.n_wavelengths))
                self.cabs = np.empty((self.n_sizes, self.n_wavelengths))
                self.csca = np.empty((self.n_sizes, self.n_wavelengths))
                self.scat_g = np.empty((self.n_sizes, self.n_wavelengths))

                self.ISRF_field_strengths = []
                for tcomment in t.meta["comments"]:
                    if "Scales" in tcomment:
                        scales = tcomment.split(":")[1].strip()
                        for number in scales.split():
                            self.ISRF_field_strengths.append(float(number))

                self.n_ISRF_strengths = len(self.ISRF_field_strengths)
                self.emission = np.empty(
                    (self.n_ISRF_strengths, self.n_sizes, self.n_wavelengths_emission)
                )

            # store the info
            self.sizes[k] = file[2]
            self.stochastic_heating[k] = stochheated
            self.cext[k, :] = t["CExt"][gindxs]
            self.csca[k, :] = t["CSca"][gindxs]
            self.cabs[k, :] = t["CAbs"][gindxs]
            self.scat_g[k, :] = t["G"][gindxs]
            if self.stochastic_heating[k]:
                base = "StEm"
                for i in range(self.n_ISRF_strengths):
                    number = str(i + 1)
                    self.emission[i, k, :] = t[base + number][egindxs]

            else:
                base = "EqEm"
                for i in range(self.n_ISRF_strengths):
                    number = str(i + 1)
                    self.emission[i, k, :] = t[base + number][egindxs]

            # convert emission from ergs/(s cm sr) to Jy/sr
            #   wavelengths in microns
            #      convert from cm^-1 to Hz^-1
            self.emission[:, k, :] *= (self.wavelengths_emission) ** 2 / 2.998e10
            self.emission[:, k, :] /= 1e-19  # convert from ergs/(s Hz) to Jy
            self.emission[:, k, :] *= 1e-6  # convert from Jy/sr to MJy/sr
            # convert from m^-2 to cm^-2
            self.emission[:, k, :] *= 1e-4

            # default size distributions
            self.size_dist[k] = self.sizes[k] ** (-4.0)

        # aliases for albedo and g calculations
        #    here they are on the same wavelength grid
        #    when calculated from an ObsData object they are not
        #    see the next function
        #    (done to allow rest of DustGrain code to be generic)
        self.n_wavelengths_scat_a = self.n_wavelengths
        self.wavelengths_scat_a = self.wavelengths
        self.scat_a_cext = self.cext
        self.scat_a_csca = self.csca

        self.n_wavelengths_scat_g = self.n_wavelengths
        self.wavelengths_scat_g = self.wavelengths
        self.scat_g_csca = self.csca

    def from_object(self, DustGrain, ObsData):
        """
        Setup a new DustGrains object on the ObsData object wavelength grids
        using an existing DustGrain object for the dust grain information.
        Currently the information is interpolated to the new wavelength grids.

        In the future, this should be enhanced to integrate across filter
        bandpasses for the data derived in filters.

        Parameters
        ----------
        DustGrain : DustGrains object
           usually read from the files with the from_files function

        ObsData: ObsData object
           contains all the observed data to be fit
        """
        self.origin = "object"

        # copy the basic information on the grain
        self.density = DustGrain.density
        self.atomic_composition = DustGrain.atomic_composition
        self.atomic_comp_names = DustGrain.atomic_comp_names
        self.atomic_comp_number = DustGrain.atomic_comp_number
        self.atomic_comp_masses = DustGrain.atomic_comp_masses
        self.mass_per_mol_comp = DustGrain.mass_per_mol_comp
        self.col_den_constant = DustGrain.col_den_constant

        self.name = DustGrain.name
        self.n_sizes = DustGrain.n_sizes
        self.sizes = DustGrain.sizes
        self.size_dist = DustGrain.size_dist
        self.stochastic_heating = DustGrain.stochastic_heating
        self.ISRF_field_strengths = DustGrain.ISRF_field_strengths
        self.n_ISRF_strengths = DustGrain.n_ISRF_strengths
        self.RF_strength = DustGrain.RF_strength

        # new values on the observed wavelength grids
        self.wavelengths = ObsData.ext_waves
        self.n_wavelengths = len(self.wavelengths)

        # variables to store the dust grain properties
        self.cext = np.empty((self.n_sizes, self.n_wavelengths))
        self.cabs = np.empty((self.n_sizes, self.n_wavelengths))
        self.csca = np.empty((self.n_sizes, self.n_wavelengths))

        self.wavelengths_emission = ObsData.ir_emission_waves
        self.n_wavelengths_emission = len(self.wavelengths_emission)
        self.emission = np.empty(
            (self.n_ISRF_strengths, self.n_sizes, self.n_wavelengths_emission)
        )

        self.wavelengths_scat_a = ObsData.scat_a_waves
        self.n_wavelengths_scat_a = len(self.wavelengths_scat_a)
        self.scat_a_cext = np.empty((self.n_sizes, self.n_wavelengths_scat_a))
        self.scat_a_csca = np.empty((self.n_sizes, self.n_wavelengths_scat_a))

        self.wavelengths_scat_g = ObsData.scat_g_waves
        self.n_wavelengths_scat_g = len(self.wavelengths_scat_g)
        self.scat_g = np.empty((self.n_sizes, self.n_wavelengths_scat_g))
        self.scat_g_csca = np.empty((self.n_sizes, self.n_wavelengths_scat_g))

        # loop over the sizes and generate grain info on the observed data grid
        for i in range(self.n_sizes):
            cext_interp = interp1d(DustGrain.wavelengths, DustGrain.cext[i, :])
            cabs_interp = interp1d(DustGrain.wavelengths, DustGrain.cabs[i, :])
            csca_interp = interp1d(DustGrain.wavelengths, DustGrain.csca[i, :])
            self.cext[i, :] = cext_interp(self.wavelengths)
            self.cabs[i, :] = cabs_interp(self.wavelengths)
            self.csca[i, :] = csca_interp(self.wavelengths)

            self.scat_a_cext[i, :] = cext_interp(self.wavelengths_scat_a)
            self.scat_a_csca[i, :] = csca_interp(self.wavelengths_scat_a)

            g_interp = interp1d(DustGrain.wavelengths, DustGrain.scat_g[i, :])
            self.scat_g[i, :] = g_interp(self.wavelengths_scat_g)
            self.scat_g_csca[i, :] = csca_interp(self.wavelengths_scat_g)

            emission_interp = interp1d(
                DustGrain.wavelengths_emission, DustGrain.emission[:, i, :]
            )
            self.emission[:, i, :] = emission_interp(self.wavelengths_emission)

    # function to integrate this component
    # returns the effective/total cabs, csca, etc.
    # these are normalized to A(V)
    def eff_grain_props(self, ObsData, predict_all=False):
        """
        Calculate the grain properties integrated over the size distribution
        for a single grain composition.

        Returns
        -------
        A dictionary of:

        C(abs) : 'numpy.ndarray' named 'cabs'
           Absorption cross section

        C(sca) : 'numpy.ndarray' named 'csca'
           Scattering cross section

        Abundances : ('list', 'numpy.ndarray') named 'natoms'
           Tuple with (atomic elements, # per/10^6 H atoms

        Emission : 'numpy.ndarray' named 'emission'
           IR emission

        albedo : 'numpy.ndarray' named 'albedo'
           Dust scattering albedo [Albedo C(sca)/Albedo C(ext)]

        g : 'numpy.ndarray' named 'g'
           Dust scattering phase function assymetry [g = <cos theta>]

        Albedo C(ext) : 'numpy.ndarray' named 'scat_a_cext'
           Extinction cross section on the albedo wavelength grid
           (needed for combining with other dust grain compositions)

        Albedo C(sca) : 'numpy.ndarray' named 'scat_a_csca'
           Scattering cross section on the albedo wavelength grid
           (needed for combining with other dust grain compositions)

        G C(sca) : 'numpy.ndarray' named 'scat_g_csca'
           Scattering cross section on the g wavelength grid
           (needed for combining with other dust grain compositions)
        """

        # output is a dictonary
        results = {}

        # initialize the results
        _effcabs = np.empty(self.n_wavelengths)
        _effcsca = np.empty(self.n_wavelengths)

        # do a very simple integration (later this could be made more complex)
        deltas = 0.5 * (self.sizes[1 : self.n_sizes] - self.sizes[0 : self.n_sizes - 1])
        sizedist1 = self.size_dist[0 : self.n_sizes - 1]
        sizedist2 = self.size_dist[1 : self.n_sizes]
        for i in range(self.n_wavelengths):
            _effcabs[i] = np.sum(
                deltas
                * (
                    (self.cabs[0 : self.n_sizes - 1, i] * sizedist1)
                    + (self.cabs[1 : self.n_sizes, i] * sizedist2)
                )
            )
            _effcsca[i] = np.sum(
                deltas
                * (
                    (self.csca[0 : self.n_sizes - 1, i] * sizedist1)
                    + (self.csca[1 : self.n_sizes, i] * sizedist2)
                )
            )

            # *not* faster to use numexpr (tested in 2015)

        results["cabs"] = _effcabs
        results["csca"] = _effcsca

        # compute the number of atoms/A(V)
        _natoms = np.empty(len(self.atomic_comp_names))
        for i in range(len(self.atomic_comp_names)):
            if self.name in [
                "a-C:H-Themis",
                "aSil-2-Themis",
            ]:  # correct for the mantles of Themis
                if self.name == "a-C:H-Themis":
                    mantle = 5 * 1e-7
                    indices = np.where(self.sizes <= mantle)[0]
                    best_index = indices[np.argmax(self.sizes[indices])] + 1
                    _natoms[i] = np.sum(
                        deltas[: best_index - 1]
                        * (
                            (
                                (self.sizes[0 : best_index - 1] ** 3)
                                * self.size_dist[0 : best_index - 1]
                                * self.col_den_constant[i]
                                * 1.6
                                / 1.3  # correcting for the densities
                            )
                            + (
                                (self.sizes[1:best_index] ** 3)
                                * self.size_dist[1:best_index]
                                * self.col_den_constant[i]
                                * 1.6
                                / 1.3
                            )
                        )
                    )
                    _natoms[i] += np.sum(
                        deltas[best_index:]
                        * (
                            (
                                (
                                    (self.sizes[best_index : self.n_sizes - 1] ** 3)
                                    - (
                                        (
                                            self.sizes[best_index : self.n_sizes - 1]
                                            - mantle
                                        )
                                        ** 3
                                    )
                                )
                                * self.size_dist[best_index : self.n_sizes - 1]
                                * self.col_den_constant[i]
                                * 1.6
                                / 1.3
                            )
                            + (
                                (
                                    (self.sizes[best_index + 1 : self.n_sizes] ** 3)
                                    - (
                                        (
                                            self.sizes[best_index + 1 : self.n_sizes]
                                            - mantle
                                        )
                                        ** 3
                                    )
                                )
                                * self.size_dist[best_index + 1 : self.n_sizes]
                                * self.col_den_constant[i]
                                * 1.6
                                / 1.3
                            )
                        )
                    )
                    _natoms[i] += np.sum(
                        deltas[best_index:]
                        * (
                            (
                                (
                                    (self.sizes[best_index : self.n_sizes - 1] - mantle)
                                    ** 3
                                )
                                * self.size_dist[best_index : self.n_sizes - 1]
                                * self.col_den_constant[i]
                            )
                            + (
                                (
                                    (self.sizes[best_index + 1 : self.n_sizes] - mantle)
                                    ** 3
                                )
                                * self.size_dist[best_index + 1 : self.n_sizes]
                                * self.col_den_constant[i]
                            )
                        )
                    )

                else:
                    mantle = 2.5 * 1e-7
                    if i == 3:
                        _natoms[i] = np.sum(
                            deltas
                            * (
                                (
                                    (
                                        (self.sizes[0 : self.n_sizes - 1] ** 3)
                                        - (
                                            (self.sizes[0 : self.n_sizes - 1] - mantle)
                                            ** 3
                                        )
                                    )
                                    * self.size_dist[0 : self.n_sizes - 1]
                                    * self.col_den_constant[i]
                                    * 1.6
                                    / 2.7  # correcting for the densities
                                )
                                + (
                                    (
                                        (self.sizes[1 : self.n_sizes] ** 3)
                                        - ((self.sizes[1 : self.n_sizes] - mantle) ** 3)
                                    )
                                    * self.size_dist[1 : self.n_sizes]
                                    * self.col_den_constant[i]
                                    * 1.6
                                    / 2.7
                                )
                            )
                        )

                    else:
                        _natoms[i] = np.sum(
                            deltas
                            * (
                                (
                                    ((self.sizes[0 : self.n_sizes - 1] - mantle) ** 3)
                                    * self.size_dist[0 : self.n_sizes - 1]
                                    * self.col_den_constant[i]
                                )
                                + (
                                    ((self.sizes[1 : self.n_sizes] - mantle) ** 3)
                                    * self.size_dist[1 : self.n_sizes]
                                    * self.col_den_constant[i]
                                )
                            )
                        )

            else:
                _natoms[i] = np.sum(
                    deltas
                    * (
                        (
                            (self.sizes[0 : self.n_sizes - 1] ** 3)
                            * self.size_dist[0 : self.n_sizes - 1]
                            * self.col_den_constant[i]
                        )
                        + (
                            (self.sizes[1 : self.n_sizes] ** 3)
                            * self.size_dist[1 : self.n_sizes]
                            * self.col_den_constant[i]
                        )
                    )
                )

        results["natoms"] = dict(zip(self.atomic_comp_names, _natoms))

        # compute the integrated emission spectrum for the right ISRF strength
        if ObsData.fit_ir_emission or predict_all:
            _emission = np.empty(self.n_wavelengths_emission)

            # Calculate the emission for the used radaiation field
            interpolated_emission = self.interpol_emission(self.RF_strength)

            for i in range(self.n_wavelengths_emission):
                _emission[i] = np.sum(
                    deltas
                    * (
                        (interpolated_emission[0 : self.n_sizes - 1, i] * sizedist1)
                        + (interpolated_emission[1 : self.n_sizes, i] * sizedist2)
                    )
                )
            results["emission"] = _emission

        # scattering parameters a & g
        if ObsData.fit_scat_a or predict_all:
            n_waves_scat_a = self.n_wavelengths_scat_a
            scat_a_cext = self.scat_a_cext
            scat_a_csca = self.scat_a_csca

            _effscat_a_cext = np.empty(n_waves_scat_a)
            _effscat_a_csca = np.empty(n_waves_scat_a)

            for i in range(n_waves_scat_a):
                _effscat_a_cext[i] = np.sum(
                    deltas
                    * (
                        (scat_a_cext[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_a_cext[1 : self.n_sizes, i] * sizedist2)
                    )
                )
                _effscat_a_csca[i] = np.sum(
                    deltas
                    * (
                        (scat_a_csca[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_a_csca[1 : self.n_sizes, i] * sizedist2)
                    )
                )

            a = []
            for i, value in enumerate(_effscat_a_cext):
                if value == 0:
                    a.append(0)
                else:
                    a.append(_effscat_a_csca[i] / value)

            results["albedo"] = np.array(a)
            results["scat_a_cext"] = _effscat_a_cext
            results["scat_a_csca"] = _effscat_a_csca

        if ObsData.fit_scat_g or predict_all:
            n_waves_scat_g = self.n_wavelengths_scat_g
            scat_g_csca = self.scat_g_csca

            _effg = np.empty(n_waves_scat_g)
            _effscat_g_csca = np.empty(n_waves_scat_g)

            for i in range(n_waves_scat_g):
                _effg[i] = np.sum(
                    deltas
                    * (
                        (
                            self.scat_g[0 : self.n_sizes - 1, i]
                            * scat_g_csca[0 : self.n_sizes - 1, i]
                            * sizedist1
                        )
                        + (
                            self.scat_g[1 : self.n_sizes, i]
                            * scat_g_csca[1 : self.n_sizes, i]
                            * sizedist2
                        )
                    )
                )
                _effscat_g_csca[i] = np.sum(
                    deltas
                    * (
                        (scat_g_csca[0 : self.n_sizes - 1, i] * sizedist1)
                        + (scat_g_csca[1 : self.n_sizes, i] * sizedist2)
                    )
                )

            g = []
            for i, value in enumerate(_effscat_g_csca):
                if value == 0:
                    g.append(0)
                else:
                    g.append(_effg[i] / value)

            results["g"] = np.array(g)
            results["scat_g_csca"] = _effscat_g_csca

        # return the results as a tuple of arrays
        return results

    def interpol_emission(self, ISRF):

        x = np.array(self.ISRF_field_strengths)
        if ISRF < x[0]:
            ISRF = x[0]
            self.RF_strength = x[0]

        elif ISRF > x[-1]:
            ISRF = x[-1]
            self.RF_strength = x[-1]

        interpolation = interp1d(x, self.emission, axis=0)
        emission = interpolation(ISRF)

        return emission
