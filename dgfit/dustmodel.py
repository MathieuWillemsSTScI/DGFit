import math
import numpy as np
from scipy.special import erf

from astropy.io import fits

from dgfit.dustgrains import DustGrains
from dgfit.dustclasses import DustCompositions

__all__ = [
    "DustModel",
    "MRN77DustModel",
    "WD01DustModel",
    "ZDA04DustModel",
    "Y24DustModel",
    "HD23DustModel",
    "Lognormal",
]


class DustModel(object):
    """
    Full dust model including arbitrary size and composition distributions.
    Includes the physical properties of the individual dust grains.

    Dust model that has each bin as an independent variable in the
    grain size distribution providing a truly arbitrary specification.

    Parameters
    ----------
    componentnames : str list, optional
        if set, then read in the grain information from files
    path : str, optional
        path to grain files
    dustmodel : DustModel object, optional
        if set, create the grain info on the obsdata wavelengths using
        the input dustmodel grain information
    obsdata : ObsData object, optional
        observed data information

    Attributes
    ----------
    origin : string
        origin of the dust grain physical properties
        allowed values are 'files' and 'onobsdata'
    n_components : int
        number of dust grain components
    components : array of DustGrain objects
        one DustGrain object per component
    sizedisttype : string
        functional form of component size distributions
    n_params : ints
        number of size distribution parameters per grain component
    parameters : dict
        Dictonary of parameters with an entry for each composition
        each entry is then a dictonary giving the value by parameter name.
        For the bins case, the dictonary is empty as the parameters is
        the size distribution.
    """

    def __init__(
        self,
        componentnames=None,
        path="./",
        dustmodel=None,
        obsdata=None,
        every_nth=5,
        limit_abundances=None,
        variable_ISRF=True,
        divide_npoints=False,
        start_ISRF=1,
        regularization=False,
        abundance_factor=False,
    ):
        self.origin = None
        self.n_components = 0
        self.components = []
        self.sizedisttype = "bins"
        self.n_params = None
        self.parameters = {}
        self.abundance_constraint = limit_abundances
        self.variable_ISRF = variable_ISRF
        self.fracs = []
        self.divide_npoints = divide_npoints
        self.start_ISRF = start_ISRF
        self.regularization = regularization
        self.abundance_factor = abundance_factor
        self.prior_ranges = {}
        self.logs = {}
        self.carbonaceous_names = DustCompositions.carbonaceous_grains
        self.silicate_names = DustCompositions.silicate_grains

        # populate the grain info
        if componentnames is not None:
            self.read_grain_files(componentnames, path=path, every_nth=every_nth)
        elif dustmodel is not None:
            if obsdata is not None:
                self.grains_on_obs(dustmodel, obsdata)
            else:
                self.grains_on_model(dustmodel)

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for component in self.components:
                n = component.n_sizes
                if self.abundance_factor:
                    n += 1
                self.n_params.append(n)

    def read_grain_files(self, componentnames, path="./", every_nth=5):
        """
        Read in the precomputed dust grain physical properties from files
        for each grain component.

        Parameters
        ----------
        componentnames : list of strings
            names of dust grain materials
        path : type
            path to files
        every_nth : int
            Only use every nth size, faster fitting

        Returns
        -------
        updated class variables
        """
        self.origin = "files"
        self.n_components = len(componentnames)
        # get the basic grain data
        for componentname in componentnames:
            cur_DG = DustGrains()
            cur_DG.from_files(componentname, path=path, every_nth=every_nth)
            self.components.append(cur_DG)

    def grains_on_obs(self, full_dustmodel, observeddata):
        """
        Calculate the dust grain properties on the observed
        wavelength grid.  Uses an existing DustModel based
        on the full precomputed files and an ObsData object
        to get the wavelength grid.  Makes the fitting faster
        to only do this transformation once.

        Parameters
        ----------
        full_dustmodel : DustModel object
            full dust model based on input files
        observeddata: ObsData object
            observed data to use for transformation

        Returns
        -------
        updated class variables
        """
        self.origin = "onobsdata"
        self.n_components = full_dustmodel.n_components
        for component in full_dustmodel.components:
            cur_DG = DustGrains()
            cur_DG.from_object(component, observeddata)
            self.components.append(cur_DG)

    def grains_on_model(self, full_dustmodel):
        """
        Calculate the dust grain properties on the model grid
        (simple copy). Uses an existing DustModel based
        on the full precomputed files and an ObsData object
        to get the wavelength grid.  Makes the fitting faster
        to only do this transformation once.

        Parameters
        ----------
        full_dustmodel : DustModel object
            full dust model based on input files

        Returns
        -------
        updated class variables
        """
        self.origin = "onmodel"
        self.n_components = full_dustmodel.n_components
        for component in full_dustmodel.components:
            self.components.append(component)

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.
        For the bins case, just passes the parameters back.  Allows for
        other functional forms of the size distribution with minimal new code.

        Parameters
        ----------
        x : floats
            grains sizes
        params : floats
            Size distribution parameters
            For the arbitrary bins case, the parameters are the number
            of grains per size distribution

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        return params

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.
        For the bins case, this does nothing.  Allows for
        other functional forms of the size distribution with minimal new code.

        Parameters
        ----------
        params : floats
            Size distribution parameters
            For the arbitrary bins case, the parameters are the number
            of grains per size distribution
        """
        pass

    def set_size_dist(self, params):
        """
        Set the size distributions for each component based on the
        parameters of the functional form of the distributions.

        Parameters
        ----------
        new_size_dists : type
            Description of parameter `new_size_dists`.

        Returns
        -------
        type
            Description of returned object.

        """
        k1 = 0
        for k, component in enumerate(self.components):
            delta_val = self.n_params[k]
            k2 = k1 + delta_val
            if self.abundance_factor:
                k2 -= 1
            component.size_dist[:] = self.compute_size_dist(
                component.sizes[:], params[k1:k2], component.name
            )
            if self.abundance_factor:
                component.abundance_factor = params[k2]
                # self.parameters[component.name][f"F_a_{k}"] = params[k2]
                k2 += 1
            if self.variable_ISRF:
                component.RF_strength = params[-1]
            k1 += delta_val

    def eff_grain_props(self, OD, predict_all=False):
        """
        Compute the effective grain properties of the ensemble of grain
        sizes and compositions.

        Parameters
        ----------
        OD : ObsData object
            Observed data object specifically used to determine which
            observations to compute (only those needed for speed)
        predict_all : type
            Regardless of the ObsData, compute all possible observations

        Returns
        -------
        dict
            Dictonary of predicted observations
            E.g., keys of cext, natoms, emission, albedo, g
        """
        # storage for results
        _cabs = np.zeros(self.components[0].n_wavelengths)
        _csca = np.zeros(self.components[0].n_wavelengths)
        _natoms = {}

        if OD.fit_ir_emission or predict_all:
            _emission = np.zeros(self.components[0].n_wavelengths_emission)

        if OD.fit_scat_a or predict_all:
            _scat_a_cext = np.zeros(self.components[0].n_wavelengths_scat_a)
            _scat_a_csca = np.zeros(self.components[0].n_wavelengths_scat_a)

        if OD.fit_scat_g or predict_all:
            _g = np.zeros(self.components[0].n_wavelengths_scat_g)
            _scat_g_csca = np.zeros(self.components[0].n_wavelengths_scat_g)

        for component in self.components:
            results = component.eff_grain_props(OD, predict_all=predict_all)

            _tcabs = results["cabs"]
            _tcsca = results["csca"]
            _cabs += _tcabs
            _csca += _tcsca

            # for the depletions (# of atoms), a bit more careful work needed
            _tnatoms = results["natoms"]
            for aname in _tnatoms.keys():
                if aname in _natoms.keys():
                    _natoms[aname] += _tnatoms[aname]
                else:
                    _natoms[aname] = _tnatoms[aname]

            if OD.fit_ir_emission or predict_all:
                _temission = results["emission"]
                _emission += _temission

            if OD.fit_scat_a or predict_all:
                _tscat_a_cext = results["scat_a_cext"]
                _tscat_a_csca = results["scat_a_csca"]
                _scat_a_cext += _tscat_a_cext
                _scat_a_csca += _tscat_a_csca

            if OD.fit_scat_g or predict_all:
                _tg = results["g"]
                _tscat_g_csca = results["scat_g_csca"]
                _g += _tscat_g_csca * _tg
                _scat_g_csca += _tscat_g_csca

        results = {}
        results["cabs"] = _cabs
        results["csca"] = _csca
        results["natoms"] = _natoms

        if OD.fit_ir_emission or predict_all:
            results["emission"] = _emission

        if OD.fit_scat_a or predict_all:
            results["albedo"] = _scat_a_csca / _scat_a_cext

        if OD.fit_scat_g or predict_all:
            results["g"] = _g / _scat_g_csca

        return results

    def read_sizedist_from_file(self, filename):
        """
        Read in the size distribution from a file interpolating
        across sizes if needed

        Parameters
        ----------
        filename : str
            name of FITS file with size distributions
            one component per extension
        """
        for k, component in enumerate(self.components):
            fitsdata = fits.getdata(filename, k + 1)

            # interpolate, otherwise assume exact match in sizes
            #   might want to add some checking here for robustness
            if len(component.size_dist) != len(fitsdata[:][1]):
                component.size_dist = 10 ** np.interp(
                    np.log10(component.sizes),
                    np.log10(fitsdata["SIZE"]),
                    np.log10(fitsdata["DIST"]),
                )
            else:
                component.size_dist = fitsdata["DIST"]

    def lnprob_generic(self, obsdata):
        """
        Compute the ln(prob) for the dust grain size and composition
        distribution as defined by the dustmodel.

        Parameters
        ----------
        obsdata : ObsData object
            All the observed data

        Returns
        -------
        float
            natural log of the probability
        """
        # get the integrated dust properties
        results = self.eff_grain_props(obsdata)

        # compute the ln(prob) for A(l)/A(V)
        lnp_alav = 0.0
        if obsdata.fit_extinction:
            cabs = results["cabs"]
            csca = results["csca"]
            cext = cabs + csca
            dust_alav = 1.086 * cext
            weights = 1.0 / (obsdata.ext_alav_unc)
            bandvals = obsdata.ext_type != "spec"
            if np.sum(bandvals) > 0:
                weights[bandvals] *= 1000
            lnp_alav = -0.5 * np.sum(((obsdata.ext_alav - dust_alav) * weights) ** 2)

        # compute the ln(prob) for the depletions
        lnp_dep = 0.0
        if obsdata.fit_abundance:
            natoms = results["natoms"]
            for atomname in natoms.keys():
                if self.abundance_constraint:
                    if (
                        natoms[atomname] - obsdata.abundance_av[atomname][0]
                        > obsdata.abundance_av[atomname][1]
                    ):
                        lnp_dep += np.inf
                    else:
                        lnp_dep += (
                            (natoms[atomname] - obsdata.abundance_av[atomname][0])
                            / (obsdata.abundance_av[atomname][1])
                        ) ** 2
                else:
                    lnp_dep += (
                        (natoms[atomname] - obsdata.abundance_av[atomname][0])
                        / (obsdata.abundance_av[atomname][1])
                    ) ** 2
        lnp_dep *= -0.5

        # compute the ln(prob) for IR emission
        lnp_emission = 0.0
        if obsdata.fit_ir_emission:
            emission = results["emission"]
            lnp_emission = -0.5 * np.sum(
                (
                    ((obsdata.ir_emission_av - emission) / (obsdata.ir_emission_av_unc))
                    ** 2
                )
            )

        # compute the ln(prob) for the dust albedo
        lnp_albedo = 0.0
        if obsdata.fit_scat_a:
            albedo = results["albedo"]
            lnp_albedo = -0.5 * np.sum(
                (((obsdata.scat_albedo - albedo) / (obsdata.scat_albedo_unc)) ** 2)
            )

        # compute the ln(prob) for the dust g
        lnp_g = 0.0
        if obsdata.fit_scat_g:
            g = results["g"]
            lnp_g = -0.5 * np.sum((((obsdata.scat_g - g) / (obsdata.scat_g_unc)) ** 2))

        total_points = (
            obsdata.ext_npts
            + obsdata.abundance_npts
            + obsdata.ir_emission_npts
            + obsdata.scat_a_npts
            + obsdata.scat_g_npts
        )

        if self.divide_npoints:
            tot = 0
            if obsdata.ext_npts > 0:
                lnp_alav -= np.log(obsdata.ext_npts)
                tot += 1
            if obsdata.abundance_npts > 0:
                lnp_dep -= np.log(obsdata.abundance_npts)
                tot += 1
            if obsdata.ir_emission_npts > 0:
                lnp_emission -= np.log(obsdata.ir_emission_npts)
                tot += 1
            if obsdata.scat_a_npts > 0:
                lnp_albedo -= np.log(obsdata.scat_a_npts)
                tot += 1
            if obsdata.scat_g_npts > 0:
                lnp_g -= np.log(obsdata.scat_g_npts)
                tot += 1
            total_points = tot

        lnp = lnp_alav + lnp_dep + lnp_emission + lnp_albedo + lnp_g
        if math.isinf(lnp) | math.isnan(lnp):
            return -np.inf

        fit_weights = [
            lnp_alav,
            lnp_dep,
            lnp_emission,
            lnp_albedo,
            lnp_g,
            lnp,
            total_points,
        ]
        self.fracs = fit_weights
        return lnp

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the full probability function including priors
        Static function as it will be called form the fitter

        Parameters
        ----------
        params : floats
            Parameters of the size distribution function
        obsdata : ObsData object
            Observed data to be fit
        dustmodel : DustModel object
            Dust model information

        Returns
        -------
        float
            natural log of the probability
        """
        # prior
        # make sure the size distributions are all positve
        lnp_bound = 0.0
        lnp_reg = 0.0

        for param in params:
            if param < 0.0:
                lnp_bound = -np.inf

        if dustmodel.regularization:
            delta = 0
            for (
                component
            ) in (
                dustmodel.components
            ):  # regularization, puts a bound on the difference between size bins depending on the width of the bin
                n_params = len(component.sizes)
                small = params[0 + delta : n_params - 1 + delta]
                big = params[1 + delta : n_params + delta]
                small_sizes = component.sizes[0 : n_params - 1]
                big_sizes = component.sizes[1:n_params]

                mask = big > 0
                small = small[mask]
                big = big[mask]
                small_sizes = small_sizes[mask]
                big_sizes = big_sizes[mask]

                nom = -(((big - small) / big) ** 2)
                denom = 2 * (((big_sizes - small_sizes) / big_sizes) ** 2)
                reg = np.sum(nom / denom)
                lnp_reg += reg
                lnp_reg /= 10
                delta += n_params

        # update the size distributions
        dustmodel.set_size_dist(params)
        return dustmodel.lnprob_generic(obsdata) + lnp_bound + lnp_reg

    def initial_walkers(self, p0, nwalkers):
        """
        Setup the walkers based on the initial parameters p0
        Specific to MCMC fitters (e.g., emcee).

        Parameters
        ----------
        p0 : floats
            Initial values of the parameters
        nwalkers : int
            Number of walkers to initialize

        Returns
        -------
        array of floats
            concatenated set of initial walker positions
        """
        self.ndim = len(p0)
        self.nwalkers = nwalkers
        # some parameters are negative, so need to be handled
        psigns = np.sign(p0)
        p = [
            psigns
            * (
                10
                ** (
                    np.log10(np.absolute(p0))
                    + 0.1 * np.random.uniform(-1, 1.0, self.ndim)
                )
            )
            for k in range(self.nwalkers)
        ]

        return p

    def save_results(self, filename, OD, size_dist_uncs=[0]):
        """
        Save fitting results to a file.  Results include the
        size distribution and all predicted observations.

        Creates a FITS file with the results

        Parameters
        ----------
        filename : str
            Name of the file to save the results
        OD : ObsData object
            All the observed data (may not be needed)
        size_dist_uncs : floats
            Uncertainties on the size distributions
        """
        # write a small primary header
        pheader = fits.Header()
        pheader.set("NCOMPS", len(self.components), "number of dust grain components")
        for k, component in enumerate(self.components):
            pheader.set(
                "CNAME" + str(k), component.name, "name of dust grain component"
            )
        pheader.set("SDMODEL", self.sizedisttype, "type of size  distribution")
        pheader.add_comment("Dust Model results written by DustModel.py")
        pheader.add_comment("written by Karl D. Gordon")
        pheader.add_comment("kgordon@stsci.edu")
        phdu = fits.PrimaryHDU(header=pheader)

        hdulist = fits.HDUList([phdu])

        # output the dust grain size distribution
        k1 = 0
        for component in self.components:
            col1 = fits.Column(name="SIZE", format="E", array=component.sizes)
            col2 = fits.Column(name="DIST", format="E", array=(component.size_dist))
            all_cols = [col1, col2]

            k2 = k1 + component.n_sizes
            if len(size_dist_uncs) > 1:
                col3 = fits.Column(
                    name="DISTPUNC", format="E", array=size_dist_uncs[0][k1:k2]
                )
                all_cols.append(col3)
                col4 = fits.Column(
                    name="DISTMUNC", format="E", array=size_dist_uncs[1][k1:k2]
                )
                all_cols.append(col4)
            k1 += component.n_sizes

            tbhdu = fits.BinTableHDU.from_columns(all_cols)
            tbhdu.header.set("EXTNAME", component.name, "dust grain component name")

            # save the parameter values
            if self.parameters:
                for cparam in self.parameters[component.name].items():
                    tbhdu.header.set(
                        cparam[0], cparam[1], "parameters of size distribution model"
                    )
            if self.abundance_factor:
                for i, component in enumerate(self.components):
                    tbhdu.header.set(
                        f"F_a_{i}",
                        1 / component.abundance_factor,
                        "parameters of abundance factor",
                    )

            hdulist.append(tbhdu)

        # output the resulting observable parameters
        results = self.eff_grain_props(OD, predict_all=True)
        cabs = results["cabs"]
        csca = results["csca"]
        natoms = results["natoms"]

        # natoms
        col1 = fits.Column(
            name="NAME", format="A2", array=np.array(list(natoms.keys()))
        )
        col2 = fits.Column(
            name="ABUND", format="E", array=np.array(list(natoms.values()))
        )
        cols = fits.ColDefs([col1, col2])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set("EXTNAME", "Abundances", "abundances in units of # atoms/A(V)")
        hdulist.append(tbhdu)

        # extinction
        col1 = fits.Column(
            name="WAVE", format="E", array=self.components[0].wavelengths
        )
        col2 = fits.Column(name="EXT", format="E", array=1.086 * (cabs + csca))
        all_cols_ext = [col1, col2]

        # emission
        emission = results["emission"]
        col1 = fits.Column(
            name="WAVE", format="E", array=self.components[0].wavelengths_emission
        )
        col2 = fits.Column(name="EMIS", format="E", array=emission)
        all_cols_emis = [col1, col2]

        # albedo
        albedo = results["albedo"]
        tvals = self.components[0].wavelengths_scat_a
        col1 = fits.Column(name="WAVE", format="E", array=tvals)
        col2 = fits.Column(name="ALBEDO", format="E", array=albedo)
        all_cols_albedo = [col1, col2]

        # g
        g = results["g"]
        tvals = self.components[0].wavelengths_scat_g
        col1 = fits.Column(name="WAVE", format="E", array=tvals)
        col2 = fits.Column(name="G", format="E", array=g)
        all_cols_g = [col1, col2]

        for k, component in enumerate(self.components):
            results = component.eff_grain_props(OD, predict_all=True)
            result_contribution = component.calculate_contribution_per_size(
                OD, predict_all=True
            )
            tcabs = results["cabs"]
            tcsca = results["csca"]
            cabs_per_size = result_contribution["cabs"]
            csca_per_size = result_contribution["csca"]
            ext_per_size = 1.086 * (cabs_per_size + csca_per_size)

            tcol = fits.Column(
                name="EXT" + str(k + 1), format="E", array=1.086 * (tcabs + tcsca)
            )
            all_cols_ext.append(tcol)

            temission = results["emission"]
            tcol = fits.Column(name="EMIS" + str(k + 1), format="E", array=temission)
            all_cols_emis.append(tcol)

            talbedo = results["albedo"]
            tcol = fits.Column(name="ALBEDO" + str(k + 1), format="E", array=talbedo)
            all_cols_albedo.append(tcol)

            tg = results["g"]
            tcol = fits.Column(name="G" + str(k + 1), format="E", array=tg)
            all_cols_g.append(tcol)

            ISRF = component.RF_strength

            hdu = fits.ImageHDU(ext_per_size)
            hdu.header["EXTNAME"] = f"EXTSIZE_{component.name}"
            hdulist.append(hdu)
            meta_cols = [
                fits.Column(name="SIZE", format="E", array=component.sizes),
                fits.Column(name="WAVE", format="E", array=component.wavelengths),
            ]
            meta_hdu = fits.BinTableHDU.from_columns(meta_cols)
            meta_hdu.header["EXTNAME"] = f"EXTSIZE_META_{component.name}"
            hdulist.append(meta_hdu)

        # now output the results
        #    extinction
        cols = fits.ColDefs(all_cols_ext)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set("EXTNAME", "Extinction", "extinction in A(lambda)/A(V)")
        hdulist.append(tbhdu)

        #    emission
        cols = fits.ColDefs(all_cols_emis)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set("EXTNAME", "Emission", "emission MJy/sr/A(V)")
        tbhdu.header.set("ISRF", ISRF, "The ISRF strength")
        hdulist.append(tbhdu)

        #    albedo
        cols = fits.ColDefs(all_cols_albedo)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set("EXTNAME", "Albedo", "dust scattering albedo")
        hdulist.append(tbhdu)

        #    g
        cols = fits.ColDefs(all_cols_g)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.header.set("EXTNAME", "G", "dust scattering phase function asymmetry")
        hdulist.append(tbhdu)

        hdulist.writeto(filename, overwrite=True)

    @staticmethod
    def get_percentile_vals(chain, ndim):
        """
        Compute the 50% +/- 33% values from the samples

        Parameters
        ----------
        chain : sampler.chain
            Chain from the EMCEE sampler
        ndim : int
            number of paramaters

        Returns
        -------
        tuple of floats
            (p50, p84-p50, p50-p16)
        """
        samples = chain.reshape((-1, ndim))
        values = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)),
        )
        val_50p, punc, munc = zip(*values)
        return (val_50p, punc, munc)

    def save_50percentile_results(
        self, oname, sampler, obsdata, nburn=0, cur_step=None
    ):
        """
        Compute the 50th percentile paramaters, set the size
        distribution, and save the results

        Creates a FITS file with the results

        Parameters
        ----------
        oname : str
            Name of the file to save the results
        sampler : emcee.sampler
            Sampler object from EMCEE run
        obsdata : ObsData object
            All the observed data (may not be needed)
        cur_step : int
            Current step number
        """
        if cur_step is None:
            cur_step = sampler.chain.shape[1]
        (
            fin_size_dist_50p,
            fin_size_dist_punc,
            fin_size_dist_munc,
        ) = self.get_percentile_vals(
            sampler.chain[:, nburn : cur_step + 1, :], self.ndim
        )
        self.set_size_dist(fin_size_dist_50p)

        # save the model parameters for the size distribution
        # set here so that the saved results have the right info
        self.set_size_dist_parameters(fin_size_dist_50p)

        # save the final size distributions
        self.save_results(
            oname, obsdata, size_dist_uncs=[fin_size_dist_punc, fin_size_dist_munc]
        )

    def save_best_results(self, oname, sampler, obsdata, cur_step=None):
        """
        Compute the best fit paramaters using a sampler chain, set the size
        distribution, and save the results

        Creates a FITS file with the results

        Parameters
        ----------
        oname : str
            Name of the file to save the results
        sampler : emcee.sampler
            Sampler object from EMCEE run
        obsdata : ObsData object
            All the observed data (may not be needed)
        cur_step : int
            Current step number
        """
        # get the best fit values
        max_lnp = -1e20
        if cur_step is None:
            cur_step = len(sampler.lnprobability[0])
        for k in range(self.nwalkers):
            tmax_lnp = np.max(sampler.lnprobability[k, 0:cur_step])
            if tmax_lnp > max_lnp:
                max_lnp = tmax_lnp
                (indxs,) = np.where(sampler.lnprobability[k] == tmax_lnp)
                fit_params_best = sampler.chain[k, indxs[0], :]

        self.set_size_dist(fit_params_best)

        # save the best fit size distributions
        self.save_results(oname, obsdata)

    def calculate_priors(self, component, obsdata):
        sizes = component.sizes
        n_sizes = len(sizes)
        col_dens = component.col_den_constant
        if component.name in self.carbonaceous_names:
            n_atoms = obsdata.abundance_av["C"][0] + obsdata.abundance_av["C"][1]
            atoms_per_grain = col_dens * (sizes**3)
        elif component.name in self.silicate_names:
            n_atoms = obsdata.abundance_av["O"][0] + obsdata.abundance_av["O"][1]
            atoms_per_grain = col_dens[3] * (sizes**3)
        n_grains = n_atoms / atoms_per_grain
        delta = sizes[1:n_sizes] - sizes[0 : n_sizes - 1]
        delta = np.append(delta, delta[-1])
        n_grains /= delta / 2

        return n_grains


# ================================================================


class MRN77DustModel(DustModel):
    """
    Dust model that uses powerlaw size distributions with min/max
    sizes (MRN).

    Same keywords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "MRN77"
        self.n_params = [4] * self.n_components
        for i, component in enumerate(self.components):
            self.parameters[component.name] = {
                "C": 1e-25 / (3.782e-22),
                "alpha": 3.5,
                "a_min": 1e-7,
                "a_max": 1e-3,
            }
            self.prior_ranges[component.name] = {
                "C": [5e-6, 1e-3],
                "alpha": [1.0, 7.0],
                "a_min": [3e-8, 1e-6],
                "a_max": [1e-4, 1e-2],
            }
            self.logs[component.name] = {
                "C": True,
                "alpha": False,
                "a_min": False,
                "a_max": False,
            }
            if self.abundance_factor:
                self.n_params[i] += 1
                self.parameters[component.name][f"F_a_{i}"] = 1

        if self.variable_ISRF:
            self.n_params.append(1)
            self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.
        Powerlaw size distribution (aka MRN size distribution)

        sizedist = A*a^-alpha

        where
            a = grain size,
            A = amplitude,
            alpha = exponent of power law,
            amin = min grain size,
            amax = max grain size,

        Parameters
        ----------
        x : floats
            grains sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        sizedist = params[0] * np.power(x, -1.0 * params[1])
        (indxs,) = np.where(np.logical_or(x < params[2], x > params[3]))
        if len(indxs) > 0:
            sizedist[indxs] = 0.0

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            self.parameters[component.name] = {
                "C": cparams[0],
                "alpha": cparams[1],
                "a_min": cparams[2],
                "a_max": cparams[3],
            }
            if self.abundance_factor:
                self.parameters[component.name][f"F_a_{k}"] = cparams[4]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        MRN model paramters for each component are
            A = amplitude
            alpha = negative of the power law exponent
            amin = min grain size
            amax = max grain size

        Parameters
        ----------
        params : array of floats 4 x n_components
            parameters of the MRN model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            # check that amin < amax (params 3 & 4)
            if cparams[2] > cparams[3]:
                lnp_bound = -np.inf

            # check that the amin and amax are within the bounds
            # of the dustmodel
            if cparams[2] < component.sizes[0]:
                lnp_bound = -np.inf
            if cparams[3] > component.sizes[-1]:
                lnp_bound = -np.inf

            # keep the normalization always positive
            if cparams[0] < 0.0:
                lnp_bound = -np.inf
            if cparams[1] < 0.0:
                lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)

            return dustmodel.lnprob_generic(obsdata) + lnp_bound


# ================================================================


class WD01DustModel(DustModel):
    """
    Dust model that uses the Weingartner & Draine (2001) size distributions.

    Same kewyords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "WD01"

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for i, component in enumerate(self.components):
                if component.name == "Silicates-DL84":
                    n = 4
                    self.parameters[component.name] = {
                        "C_s": 1.33e-12 / (5.345e-22),
                        "a_ts": 0.171e4,
                        "alpha_s": -1.41,
                        "beta_s": -11.5,
                    }
                    self.prior_ranges[component.name] = {
                        "C_s": [1e7, 2e11],
                        "a_ts": [500, 3000],
                        "alpha_s": [-4, 1],
                        "beta_s": [-100, 0],
                    }
                    self.logs[component.name] = {
                        "C_s": True,
                        "a_ts": True,
                        "alpha_s": False,
                        "beta_s": False,
                    }

                elif component.name == "Carbonaceous-LD01":
                    n = 6
                    self.parameters[component.name] = {
                        "C_g": 4.15e-11 / (5.345e-22),
                        "a_tg": 0.00837e4,
                        "alpha_g": -1.91,
                        "beta_g": -0.125,
                        "a_cg": 0.499e4,
                        "b_C": 3.0e-5 / (5.345e-22),
                    }
                    self.prior_ranges[component.name] = {
                        "C_g": [1e7, 1e12],
                        "a_tg": [1, 3000],
                        "alpha_g": [-4, 10],
                        "beta_g": [-10, 6],
                        "a_cg": [50, 3000],
                        "b_C": [1e16, 1e17],
                    }
                    self.logs[component.name] = {
                        "C_g": True,
                        "a_tg": True,
                        "alpha_g": False,
                        "beta_g": False,
                        "a_cg": True,
                        "b_C": True,
                    }
                else:
                    raise ValueError(
                        "%s grain material note supported" % component.name
                    )

                if self.abundance_factor:
                    n += 1
                    self.parameters[component.name][f"F_a_{i}"] = 1
                self.n_params.append(n)

            if self.variable_ISRF:
                self.n_params.append(1)
                self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.

        Parameters
        ----------
        x : floats
            grain sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        # input grain sizes are in cm, needed in Angstroms
        a = x * 1e8

        if len(params) == 6:
            # carbonaceous
            C, a_t, alpha, beta, a_c, input_bC = params[:6]
        else:
            # silicates
            C, a_t, alpha, beta = params[:4]
            a_c = 0.1e4
            input_bC = None

        # larger grain size distribution
        # same for silicates and carbonaceous grains
        if beta >= 0.0:
            Fa = 1.0 + (beta * a / a_t)
        else:
            Fa = 1.0 / (1.0 - (beta * a / a_t))

        Ga = np.full((len(a)), 1.0)
        (indxs,) = np.where(a > a_t)
        Ga[indxs] = np.exp(-1.0 * np.power((a[indxs] - a_t) / a_c, 3.0))

        sizedist = (C / (1e-8 * a)) * np.power(a / a_t, alpha) * Fa * Ga

        # very small gain size distribution
        # only for carbonaceous grains
        if input_bC is not None:
            a0 = np.array([3.5, 30.0])  # in A
            bC = np.array([0.75, 0.25]) * input_bC
            sigma = 0.4
            rho = 2.24  # in g/cm^3 for graphite
            mC = 12.0107 * 1.660e-24

            Da = 0.0
            for i in range(2):
                Bi = (
                    (3.0 / (np.power(2.0 * np.pi, 1.5)))
                    * (
                        np.exp(-4.5 * np.power(sigma, 2.0))
                        / (rho * np.power(1e-8 * a0[i], 3.0) * sigma)
                    )
                    * (
                        bC[i]
                        * mC
                        / (
                            1.0
                            + erf(
                                (3.0 * sigma / np.sqrt(2.0))
                                + np.log(a0[i] / 3.5) / (sigma * np.sqrt(2.0))
                            )
                        )
                    )
                )

                Da += (Bi / (1e-8 * a)) * np.exp(
                    -0.5 * np.power(np.log(a / a0[i]) / sigma, 2.0)
                )

            sizedist += Da

        if composition == "Carbonaceous-LD01":
            (indxs,) = np.where(np.logical_or(a < 3.5, a > 1e4))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Silicates-DL84":
            (indxs,) = np.where(np.logical_or(a < 3.5, a > 3e3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            if component.name == "Silicates-DL84":
                self.parameters[component.name] = {
                    "C_s": cparams[0],
                    "a_ts": cparams[1],
                    "alpha_s": cparams[2],
                    "beta_s": cparams[3],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[4]

            elif component.name == "Carbonaceous-LD01":
                self.parameters[component.name] = {
                    "C_g": cparams[0],
                    "a_tg": cparams[1],
                    "alpha_g": cparams[2],
                    "beta_g": cparams[3],
                    "a_cg": cparams[4],
                    "b_C": cparams[5],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[6]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        Parameters
        ----------
        params : array of floats 4
            parameters of the WD model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            # keep the normalization always positive
            if component.name in ["Silicates-DL84", "Carbonaceous-LD01"]:
                if cparams[0] < 0.0:
                    lnp_bound = -np.inf
                if cparams[1] < 0.0:
                    lnp_bound = -np.inf
                if component.name == "Carbonaceous-LD01":
                    if cparams[4] < 0.0:
                        lnp_bound = -np.inf
                    if cparams[5] < 0.0:
                        lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound


# ================================================================


class ZDA04DustModel(DustModel):
    """
    Dust model that uses the Zubko et al. (2004) size distributions.

    Same kewyords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "ZDA04"

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for i, component in enumerate(self.components):
                if component.name == "Carbonaceous-LD01":
                    n = 7
                    self.parameters[component.name] = {  # ACH2 model /// Graphite model
                        f"A_{i}": 2.484404e-3
                        / (5.34e-22),  # 4.727727e-3 /// 2.484404e-3
                        f"c_0_{i}": -8.54571,  # -8.91244 /// -8.54571
                        f"b_0_{i}": -3.60112,  # -3.72015 /// -3.60112
                        f"b_1_{i}": 1.86525e5,  # 6.78215e5 /// 1.86525e5
                        f"m_1_{i}": -13.5755,  # -14.2532 /// -13.5755
                        f"a_3_{i}": 1.98119e-3,  # 1.58225e-3 /// 1.98119e-3
                        f"m_3_{i}": 9.25894,  # 8.71891 /// 9.25894
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e12, 1e20],
                        f"c_0_{i}": [-15, -3],
                        f"b_0_{i}": [-15, -1],
                        f"b_1_{i}": [5, 5e10],
                        f"m_1_{i}": [-50, -5],
                        f"a_3_{i}": [0.000001, 0.01],
                        f"m_3_{i}": [5, 30],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"c_0_{i}": False,
                        f"b_0_{i}": False,
                        f"b_1_{i}": True,
                        f"m_1_{i}": False,
                        f"a_3_{i}": True,
                        f"m_3_{i}": False,
                    }

                elif component.name == "Graphite-LD93":
                    n = 12
                    self.parameters[component.name] = {
                        f"A_{i}": 1.901190e-3 / (5.34e-22),
                        f"c_0_{i}": -10.1149,
                        f"b_0_{i}": -5.3308,
                        f"b_1_{i}": 7.54276e-2,
                        f"a_1_{i}": 8.08703e-2,
                        f"m_1_{i}": 3.37644,
                        f"b_3_{i}": 1.12502e3,
                        f"a_3_{i}": 0.145378,
                        f"m_3_{i}": 3.49042,
                        f"b_4_{i}": 1.12602e3,
                        f"a_4_{i}": 0.169079,
                        f"m_4_{i}": 3.636654,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e12, 1e20],
                        f"c_0_{i}": [-15, -5],
                        f"b_0_{i}": [-9, -1],
                        f"b_1_{i}": [1e-4, 1e-1],
                        f"a_1_{i}": [0.0001, 1],
                        f"m_1_{i}": [0, 10],
                        f"b_3_{i}": [1e2, 1e4],
                        f"a_3_{i}": [0.01, 1],
                        f"m_3_{i}": [0, 10],
                        f"b_4_{i}": [1e2, 1e4],
                        f"a_4_{i}": [0.01, 1],
                        f"m_4_{i}": [0, 15],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"c_0_{i}": False,
                        f"b_0_{i}": False,
                        f"b_1_{i}": False,
                        f"a_1_{i}": True,
                        f"m_1_{i}": False,
                        f"b_3_{i}": True,
                        f"a_3_{i}": True,
                        f"m_3_{i}": False,
                        f"b_4_{i}": True,
                        f"a_4_{i}": True,
                        f"m_4_{i}": False,
                    }

                elif component.name == "Silicates-DL84":
                    n = 12
                    self.parameters[component.name] = {
                        f"A_{i}": 1.541199e-3 / (5.34e-22),
                        f"c_0_{i}": -8.53081,
                        f"b_0_{i}": -3.70009,
                        f"b_1_{i}": 3.96003e-9,
                        f"a_1_{i}": 9.11246e-3,
                        f"m_1_{i}": 47.0606,
                        f"b_3_{i}": 1.48e3,
                        f"a_3_{i}": 0.484381,
                        f"m_3_{i}": 12.3253,
                        f"b_4_{i}": 1.481e3,
                        f"a_4_{i}": 0.474035,
                        f"m_4_{i}": 12.0995,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e12, 15],
                        f"c_0_{i}": [-15, 0],
                        f"b_0_{i}": [-7, -1],
                        f"b_1_{i}": [1e-12, 1e-4],
                        f"a_1_{i}": [0.0001, 0.1],
                        f"m_1_{i}": [10, 500],
                        f"b_3_{i}": [5, 5000],
                        f"a_3_{i}": [0.00001, 0.001],
                        f"m_3_{i}": [5, 100],
                        f"b_4_{i}": [10, 5000],
                        f"a_4_{i}": [0.0001, 1.0],
                        f"m_4_{i}": [0, 50],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"c_0_{i}": False,
                        f"b_0_{i}": False,
                        f"b_1_{i}": False,
                        f"a_1_{i}": True,
                        f"m_1_{i}": False,
                        f"b_3_{i}": True,
                        f"a_3_{i}": True,
                        f"m_3_{i}": False,
                        f"b_4_{i}": True,
                        f"a_4_{i}": True,
                        f"m_4_{i}": False,
                    }

                elif component.name == "amC-ACH2-Z96":
                    n = 6
                    self.parameters[component.name] = {
                        f"A_{i}": 7.862901e-8 / (5.34e-22),
                        f"c_0_{i}": -3.92513,
                        f"b_0_{i}": -3.54913,
                        f"b_1_{i}": 2.13708e-17,
                        f"a_1_{i}": 2.03908e-4,
                        f"m_1_{i}": 34.7835,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e10, 1e16],
                        f"c_0_{i}": [-6, 2],
                        f"b_0_{i}": [-6, 0],
                        f"b_1_{i}": [1e-32, 1e-20],
                        f"a_1_{i}": [0.0000001, 1e-4],
                        f"m_1_{i}": [15, 50],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"c_0_{i}": False,
                        f"b_0_{i}": False,
                        f"b_1_{i}": True,
                        f"a_1_{i}": True,
                        f"m_1_{i}": False,
                    }

                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )

                if self.abundance_factor:
                    n += 1
                    self.parameters[component.name][f"F_a_{i}"] = 1
                self.n_params.append(n)

            if self.variable_ISRF:
                self.n_params.append(1)
                self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.

        Parameters
        ----------
        x : floats
            grain sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        # input grain sizes are in cm, needed in microns
        a = x * 1e4

        if composition in ["Silicates1-ZDA04"]:
            A, c_0, b_0, b_1, a_1, m_1, b_2, a_2, m_2 = params[:9]
            # b_3 = a_3 = m_3 = b_4 = a_4 = m_4 = 0
            term3 = b_2 * (np.abs(np.log10(a / a_2)) ** m_2)
            term4 = 0.0
            term5 = 0.0

        if composition in ["Carbonaceous-LD01"]:
            A, c_0, b_0, b_1, m_1, a_3, m_3 = params[:7]
            a_1 = 1
            b_3 = 1e24
            # b_2 = a_2 = m_2 = b_4 = a_4 = m_4 = 0
            term3 = 0.0
            term4 = b_3 * np.power(np.abs(a - a_3), m_3)
            term5 = 0.0

        if composition in ["amC-ACH2-Z96"]:
            A, c_0, b_0, b_1, a_1, m_1 = params[:6]
            # b_2 = a_2 = m_2 = b_3 = a_3 = m_3 = b_4 = a_4 = m_4 = 0
            term3 = 0.0
            term4 = 0.0
            term5 = 0.0

        if composition in ["Graphite-LD93", "Silicates-DL84"]:
            A, c_0, b_0, b_1, a_1, m_1, b_3, a_3, m_3, b_4, a_4, m_4 = params[:12]
            # b_2 = a_2 = m_2 = 0
            term3 = 0.0
            term4 = b_3 * np.power(np.abs(a - a_3), m_3)
            term5 = b_4 * np.power(np.abs(a - a_4), m_4)

        term1 = b_0 * np.log10(a)
        term2 = b_1 * np.power(np.abs(np.log10(a / a_1)), m_1)

        ga = c_0 + term1 - term2 - term3 - term4 - term5

        sizedist = A * (np.float64(10) ** np.float64(ga))

        if composition == "amC-ACH2-Z96":
            (indxs,) = np.where(np.logical_or(a < 0.02, a > 0.28))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Carbonaceous-LD01":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 5e-3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Silicates-DL84":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 0.34))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Graphite-LD93":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 0.3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            if component.name == "Carbonaceous-LD01":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"c_0_{k}": cparams[1],
                    f"b_0_{k}": cparams[2],
                    f"b_1_{k}": cparams[3],
                    f"m_1_{k}": cparams[4],
                    f"a_3_{k}": cparams[5],
                    f"m_3_{k}": cparams[6],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[7]

            elif component.name == "Graphite-LD93":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"c_0_{k}": cparams[1],
                    f"b_0_{k}": cparams[2],
                    f"b_1_{k}": cparams[3],
                    f"a_1_{k}": cparams[4],
                    f"m_1_{k}": cparams[5],
                    f"b_3_{k}": cparams[6],
                    f"a_3_{k}": cparams[7],
                    f"m_3_{k}": cparams[8],
                    f"b_4_{k}": cparams[9],
                    f"a_4_{k}": cparams[10],
                    f"m_4_{k}": cparams[11],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[12]

            elif component.name == "Silicates-DL84":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"c_0_{k}": cparams[1],
                    f"b_0_{k}": cparams[2],
                    f"b_1_{k}": cparams[3],
                    f"a_1_{k}": cparams[4],
                    f"m_1_{k}": cparams[5],
                    f"b_3_{k}": cparams[6],
                    f"a_3_{k}": cparams[7],
                    f"m_3_{k}": cparams[8],
                    f"b_4_{k}": cparams[9],
                    f"a_4_{k}": cparams[10],
                    f"m_4_{k}": cparams[11],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[12]

            elif component.name == "amC-ACH2-Z96":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"c_0_{k}": cparams[1],
                    f"b_0_{k}": cparams[2],
                    f"b_1_{k}": cparams[3],
                    f"a_1_{k}": cparams[4],
                    f"m_1_{k}": cparams[5],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[6]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        Parameters
        ----------
        params : array of floats
            parameters of the Zubko model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            # keep the normalization always positive
            if cparams[0] < 0:
                lnp_bound = -np.inf
            if component.name != "Carbonaceous-LD01":
                if cparams[4] <= 0.0:
                    lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound


# ================================================================


class HD23DustModel(DustModel):
    """
    Dust model that uses the Hensley & Draine (2023) size distributions.

    Same kewyords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "HD23"

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for i, component in enumerate(self.components):
                if component.name == "Carbonaceous-DL07":
                    n = 2
                    self.parameters[component.name] = {
                        "B_1": 7.52e-7 / (3.24e-22),
                        "B_2": 8.09e-10 / (3.24e-22),
                    }
                    self.prior_ranges[component.name] = {
                        "B_1": [1e12, 1e17],
                        "B_2": [1e9, 1e14],
                    }
                    self.logs[component.name] = {
                        "B_1": True,
                        "B_2": True,
                    }

                elif component.name == "AstroDust-DH21":
                    n = 9
                    self.parameters[component.name] = {
                        "B_ad": 3.312432756747526242e-10 / (3.24e-22),
                        "a_0": 63.80845916490116565,
                        "sigma_ad": 0.3525536658924082190,
                        "A_0": 2.973514508974622639e-5 / (3.24e-22),
                        "A_1": -3.401700031709036676,
                        "A_2": -0.8070693618339355169,
                        "A_3": 0.1565691274812446021,
                        "A_4": 7.963246509606041607e-3,
                        "A_5": -1.680451603515705633e-3,
                    }
                    self.prior_ranges[component.name] = {
                        "B_ad": [1e9, 1e14],
                        "a_0": [10, 500],
                        "sigma_ad": [0.01, 1.0],
                        "A_0": [1e14, 5e18],
                        "A_1": [-10, 10],
                        "A_2": [-5, 5],
                        "A_3": [-2.5, 2.5],
                        "A_4": [-0.5, 0.5],
                        "A_5": [-0.1, 0.1],
                    }
                    self.logs[component.name] = {
                        "B_ad": True,
                        "a_0": True,
                        "sigma_ad": False,
                        "A_0": True,
                        "A_1": False,
                        "A_2": False,
                        "A_3": False,
                        "A_4": False,
                        "A_5": False,
                    }

                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )

                if self.abundance_factor:
                    n += 1
                    self.parameters[component.name][f"F_a_{i}"] = 1
                self.n_params.append(n)

            if self.variable_ISRF:
                self.n_params.append(1)
                self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.

        Parameters
        ----------
        x : floats
            grain sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        # input grain sizes are in cm, needed in angstrom
        a = x * 1e8
        sigma = 0.4

        if len(params) == 2:
            B_1, B_2 = params[:2]
            B_ad = None
            a_01 = 4.0
            a_02 = 30

        else:
            B_ad, a_0, sigma_ad = params[:3]
            A = params[3:]

        if B_ad is None:
            sizedist = (B_1 / (1e-8 * a)) * np.exp(
                -np.power(np.log(a / a_01), 2) / (2 * np.power(sigma, 2))
            ) + (B_2 / (1e-8 * a)) * np.exp(
                -np.power(np.log(a / a_02), 2) / (2 * np.power(sigma, 2))
            )

        else:
            sizedist = (B_ad / (1e-8 * a)) * np.exp(
                -(np.power(np.log(a / a_0), 2)) / (2 * np.power(sigma_ad, 2))
            )
            exponent = 0
            for i in range(5):
                exponent += A[i + 1] * (np.power(np.log(a), (i + 1)))
            sizedist += (A[0] / (1e-8 * a)) * np.exp(exponent)

        if composition == "Carbonaceous-DL07":
            (indxs,) = np.where(np.logical_or(a < 4, a > 1e3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "AstroDust-DH21":
            (indxs,) = np.where(np.logical_or(a < 4.5, a > 5e4))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            if component.name == "Carbonaceous-DL07":
                self.parameters[component.name] = {
                    "B_1": cparams[0],
                    "B_2": cparams[1],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[2]

            elif component.name == "AstroDust-DH21":
                self.parameters[component.name] = {
                    "B_ad": cparams[0],
                    "a_0": cparams[1],
                    "sigma_ad": cparams[2],
                    "A_0": cparams[3],
                    "A_1": cparams[4],
                    "A_2": cparams[5],
                    "A_3": cparams[6],
                    "A_4": cparams[7],
                    "A_5": cparams[8],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[9]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        Parameters
        ----------
        params : array of floats
            parameters of the Zubko model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            if component.name == "Carbonaceous-DL07":
                # keep the normalization always positive
                if cparams[0] < 0.0:
                    lnp_bound = -np.inf
                if cparams[1] < 0.0:
                    lnp_bound = -np.inf

            elif component.name == "AstroDust-DH21":
                if cparams[0] < 0.0:
                    lnp_bound = -np.inf
                if cparams[3] < 0.0:
                    lnp_bound = -np.inf
                if cparams[2] < 0.25:
                    lnp_bound = -np.inf
                if cparams[2] > 0.8:
                    lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound


# ================================================================


class Y24DustModel(DustModel):
    """
    Dust model that uses the Themis 2.0 (2024) size distributions.

    Same kewyords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "Y24"

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for i, component in enumerate(self.components):
                if component.name == "a-C-J16":
                    n = 5
                    self.parameters[component.name] = {
                        f"A_{i}": 1.3412712e-18 / (5.34e-22),
                        "alpha": -5,
                        "a_C": 0.05,
                        "a_t": 0.01,
                        "gamma": 1,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e2, 1e5],
                        "alpha": [-10, 0],
                        "a_C": [0.0001, 0.5],
                        "a_t": [0, 0.2],
                        "gamma": [0, 5],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        "alpha": False,
                        "a_C": False,
                        "a_t": False,
                        "gamma": False,
                    }

                elif component.name == "a-C:H-J16":
                    n = 3
                    self.parameters[component.name] = {
                        f"A_{i}": 3.1841239e-9 / (5.34e-22),
                        f"a_0_{i}": 6.195341e-3,
                        f"sigma_{i}": 1.315171,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e10, 1e14],
                        f"a_0_{i}": [0, 0.5],
                        f"sigma_{i}": [0.5, 5.0],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"a_0_{i}": False,
                        f"sigma_{i}": False,
                    }

                elif component.name == "X35-X50B-D22":
                    n = 3
                    self.parameters[component.name] = {
                        f"A_{i}": 6.9843816e-6 / (5.34e-22),
                        f"a_0_{i}": 9.210816e-04,
                        f"sigma_{i}": 1.217290,
                    }
                    self.prior_ranges[component.name] = {
                        f"A_{i}": [1e14, 1e18],
                        f"a_0_{i}": [0, 0.5],
                        f"sigma_{i}": [0.5, 5.0],
                    }
                    self.logs[component.name] = {
                        f"A_{i}": True,
                        f"a_0_{i}": False,
                        f"sigma_{i}": False,
                    }

                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )

                if self.abundance_factor:
                    n += 1
                    self.parameters[component.name][f"F_a_{i}"] = 1
                self.n_params.append(n)

            if self.variable_ISRF:
                self.n_params.append(1)
                self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.

        Parameters
        ----------
        x : floats
            grain sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        # input grain sizes are in cm, needed in um
        a = x * 1e4

        if len(params) == 5:
            A, alpha, a_C, a_t, gamma = params[:5]
            sigma = None

        else:
            A, a_0, sigma = params[:3]

        if sigma is not None:
            sizedist = (A / a) * np.exp(
                -np.power(np.log(a / a_0), 2) / (2 * np.power(sigma, 2))
            )

        else:
            small_indices = a <= a_t
            large_indices = a > a_t
            small = (A / (a[small_indices])) * np.power(a[small_indices], alpha)
            large = (
                (A / (a[large_indices]))
                * np.power(a[large_indices], alpha)
                * np.exp(-np.power((a[large_indices] - a_t) / a_C, gamma))
            )

            sizedist = np.concatenate((small, large))

        if composition == "a-C:H-J16":
            (indxs,) = np.where(np.logical_or(a < 0.04495579, a > 0.7))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "X35-X50B-D22":
            (indxs,) = np.where(np.logical_or(a < 0.011, a > 0.3737511))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "a-C-J16":
            (indxs,) = np.where(np.logical_or(a < 0.0004, a > 0.025))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            if component.name == "a-C-J16":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    "alpha": cparams[1],
                    "a_C": cparams[2],
                    "a_t": cparams[3],
                    "gamma": cparams[4],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[5]

            elif component.name == "a-C:H-J16":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"a_0_{k}": cparams[1],
                    f"sigma_{k}": cparams[2],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[3]

            elif component.name == "X35-X50B-D22":
                self.parameters[component.name] = {
                    f"A_{k}": cparams[0],
                    f"a_0_{k}": cparams[1],
                    f"sigma_{k}": cparams[2],
                }
                if self.abundance_factor:
                    self.parameters[component.name][f"F_a_{k}"] = cparams[3]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        Parameters
        ----------
        params : array of floats
            parameters of the Themis model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            # keep the normalization always positive
            if cparams[0] < 0.0:
                lnp_bound = -np.inf

            if component.name == "a-C-J16":
                if cparams[2] <= 0:
                    lnp_bound = -np.inf
                if cparams[3] <= 0:
                    lnp_bound = -np.inf

            elif component.name in ["X35-X50B-D22", "a-C:H-J16"]:
                if cparams[1] <= 0:
                    lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound


class Lognormal(DustModel):
    """
    Dust model that uses lognormals for the size distributions.

    Same kewyords and attributes as the parent DustModel class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizedisttype = "lognormals"

        # set the number of size distribution parametres
        if self.n_components > 0:
            self.n_params = []
            for i, component in enumerate(self.components):
                n = 6
                sizes = (
                    component.sizes * 1e4
                )  # input grain sizes are in cm, needed in microns
                n_sizes = len(sizes)
                middle = n_sizes // 2

                self.parameters[component.name] = {
                    f"A_s_{i}": 1e19,
                    f"sigma_s_{i}": 0.5,
                    f"a_s_{i}": sizes[3],
                    f"A_b_{i}": 7e10,
                    f"sigma_b_{i}": 0.5,
                    f"a_b_{i}": sizes[middle],
                }
                self.prior_ranges[component.name] = {
                    f"A_s_{i}": [5e14, 5e21],
                    f"sigma_s_{i}": [0, 10.0],
                    f"a_s_{i}": [sizes[0], sizes[20]],
                    f"A_b_{i}": [1e4, 1e12],
                    f"sigma_b_{i}": [0, 30],
                    f"a_b_{i}": [sizes[0], sizes[-1]],
                }
                self.logs[component.name] = {
                    f"A_s_{i}": True,
                    f"sigma_s_{i}": False,
                    f"a_s_{i}": True,
                    f"A_b_{i}": True,
                    f"sigma_b_{i}": False,
                    f"a_b_{i}": True,
                }
                if self.abundance_factor:
                    n += 1
                    self.parameters[component.name][f"F_a_{i}"] = 1
                self.n_params.append(n)

            if self.variable_ISRF:
                self.n_params.append(1)
                self.parameters["Radiation field"] = {"RF": self.start_ISRF}

    def compute_size_dist(self, x, params, composition):
        """
        Compute the size distribution for the input sizes.

        Parameters
        ----------
        x : floats
            grain sizes
        params : floats
            Size distribution parameters

        Returns
        -------
        floats
            Size distribution as a function of x
        """
        # input grain sizes are in cm, needed in um
        a = x * 1e4

        A_s, sigma_s, a_s, A_b, sigma_b, a_b = params[:6]

        sizedist = (A_s / (sigma_s * a)) * np.exp(
            -np.power(np.log(a / a_s), 2) / (2 * np.power(sigma_s, 2))
        ) + (A_b / (sigma_b * a)) * np.exp(
            -np.power(np.log(a / a_b), 2) / (2 * np.power(sigma_b, 2))
        )

        return sizedist

    def set_size_dist_parameters(self, params):
        """
        Set the size distribution parameters in the object dictonary.

        Parameters
        ----------
        params : floats
            Size distribution parameters
        """
        k1 = 0
        for k, component in enumerate(self.components):
            k2 = k1 + self.n_params[k]
            cparams = params[k1:k2]
            k1 += self.n_params[k]
            self.parameters[component.name] = {
                f"A_s_{k}": cparams[0],
                f"sigma_s_{k}": cparams[1],
                f"a_s_{k}": cparams[2],
                f"A_b_{k}": cparams[3],
                f"sigma_b_{k}": cparams[4],
                f"a_b_{k}": cparams[5],
            }
            if self.abundance_factor:
                self.parameters[component.name][f"F_a_{k}"] = cparams[6]

        if self.variable_ISRF:
            self.parameters["Radiation field"] = {"RF": params[-1]}

    @staticmethod
    def lnprob(params, obsdata, dustmodel):
        """
        Compute the ln(prob) given the model parameters

        Parameters
        ----------
        params : array of floats
            parameters of the Themis model
        obsdata : ObsData object
            observed data for fitting
        dustmodel : DustModel object
            must be passed explicitly as the fitters
            require a static method (is this true?)

        Returns
        -------
        lnprob : float
            natural log of the probability the input parameters
            describe the data
        """
        # priors
        k1 = 0
        lnp_bound = 0.0
        for k, component in enumerate(dustmodel.components):
            # get the parameters for the current component
            k2 = k1 + dustmodel.n_params[k]
            cparams = params[k1:k2]
            k1 += dustmodel.n_params[k]

            # keep the normalization always positive
            if any(cparams[i] < 0.0 for i in range(0, 6)):
                lnp_bound = -np.inf

            if params[5] <= params[2]:
                lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound
