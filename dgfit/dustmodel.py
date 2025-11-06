import math
import numpy as np
from scipy.special import erf

from astropy.io import fits

from dgfit.dustgrains import DustGrains

__all__ = [
    "DustModel",
    "MRN77DustModel",
    "WD01DustModel",
    "ZDA04DustModel",
    "Y24DustModel",
    "HD23DustModel",
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
        regularization=False
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
        self.regularization=regularization

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
                self.n_params.append(component.n_sizes)

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
            component.size_dist[:] = self.compute_size_dist(
                component.sizes[:], params[k1:k2], component.name
            )
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
                if natoms[atomname] < obsdata.abundance_av[atomname][0]:
                    lnp_dep += 0.0
                else:
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
                lnp_alav /= obsdata.ext_npts
                tot += 1
            if obsdata.abundance_npts > 0:
                lnp_dep /= obsdata.abundance_npts
                tot += 1
            if obsdata.ir_emission_npts > 0:
                lnp_emission /= obsdata.ir_emission_npts
                tot += 1
            if obsdata.scat_a_npts > 0:
                lnp_albedo /= obsdata.scat_a_npts
                tot += 1
            if obsdata.scat_g_npts:
                lnp_g /= obsdata.scat_g_npts
                tot += 1
            total_points = tot

        # combine the lnps
        lnp = lnp_alav + lnp_dep + lnp_emission + lnp_albedo + lnp_g
        fit_weights = [
            lnp_alav / lnp,
            lnp_dep / lnp,
            lnp_emission / lnp,
            lnp_albedo / lnp,
            lnp_g / lnp,
            total_points,
        ]
        self.fracs = fit_weights

        if math.isinf(lnp) | math.isnan(lnp):
            return -np.inf
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
        #    make sure the size distributions are all positve
        lnp_bound = 0.0
        lnp_reg = 0.0

        for param in params:
            if param < 0.0:
                lnp_bound = -np.inf

        if dustmodel.regularization:
            delta = 0
            for component in dustmodel.components:
                n_params = len(component.sizes)
                small = params[0 + delta : n_params - 1 + delta]
                big = params[1 + delta : n_params + delta]
                small_sizes = component.sizes[0 : n_params - 1]
                big_sizes = component.sizes[1 : n_params]
                nom = -(((big - small)/ big) ** 2)
                denom = (2 * (((big_sizes - small_sizes) / big_sizes)**2))
                reg = np.sum(nom/denom)
                lnp_reg += reg
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
            tcabs = results["cabs"]
            tcsca = results["csca"]
            # tnatoms = results['natoms']

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
        self.n_params.append(1)
        for component in self.components:
            self.parameters[component.name] = {
                "C": 1e-25 / (3.782e-22),
                "alpha": 3.5,
                "a_min": 1e-7,
                "a_max": 1e-3,
            }
        if self.variable_ISRF:
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
        
        if not (0.25 <= params[-1] <= 20):
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
            for component in self.components:
                if component.name == "astro-silicates-WD01":
                    self.n_params.append(4)
                    self.parameters["astro-silicates-WD01"] = {
                        "C_s": 1.33e-12 / (5.345e-22),
                        "a_ts": 0.171e4,
                        "alpha_s": -1.41,
                        "beta_s": -11.5,
                    }
                elif component.name == "astro-carbonaceous-WD01":
                    self.n_params.append(6)
                    self.parameters["astro-carbonaceous-WD01"] = {
                        "C_g": 4.15e-11 / (5.345e-22),
                        "a_tg": 0.00837e4,
                        "alpha_g": -1.91,
                        "beta_g": -0.125,
                        "a_cg": 0.499e4,
                        "b_C": 3.0e-5 / (5.345e-22),
                    }
                else:
                    raise ValueError(
                        "%s grain material note supported" % component.name
                    )
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
            C, a_t, alpha, beta, a_c, input_bC = params
        else:
            # silicates
            C, a_t, alpha, beta = params
            a_c = 0.1e4
            input_bC = None

        # larger grain size distribution
        # same for silicates and carbonaceous grains
        if beta >= 0.0:
            Fa = 1.0 + beta * a / a_t
        else:
            Fa = 1.0 / (1.0 - beta * a / a_t)

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

        if composition == "astro-carbonaceous-WD01":
            (indxs,) = np.where(np.logical_or(a < 3.5, a > 1e4))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "astro-silicates-WD01":
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
            if component.name == "astro-silicates-WD01":
                self.parameters["astro-silicates-WD01"] = {
                    "C_s": cparams[0],
                    "a_ts": cparams[1],
                    "alpha_s": cparams[2],
                    "beta_s": cparams[3],
                }
            elif component.name == "astro-carbonaceous-WD01":
                self.parameters["astro-carbonaceous-WD01"] = {
                    "C_g": cparams[0],
                    "a_tg": cparams[1],
                    "alpha_g": cparams[2],
                    "beta_g": cparams[3],
                    "a_cg": cparams[4],
                    "b_C": cparams[5],
                }
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
            if component.name in ["astro-silicates-WD01", "astro-carbonaceous-WD01"]:
                if cparams[0] < 0.0:
                    lnp_bound = -1e20
                if cparams[1] < 0.0:
                    lnp_bound = -1e20
                if component.name == "astro-carbonaceous-WD01":
                    if cparams[4] < 0.0:
                        lnp_bound = -1e20
                    if cparams[5] < 0.0:
                        lnp_bound = -1e20

        if not (0.25 <= params[-1] <= 20):
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
            for component in self.components:
                if component.name == "PAH-ZDA04":
                    self.n_params.append(7)
                    self.parameters["PAH-ZDA04"] = {  # ACH2 model /// Graphite model
                        "A": 2.484404e-3 / (5.34e-22),  # 4.727727e-3 /// 2.484404e-3
                        "c_0": -8.54571,  # -8.91244 /// -8.54571
                        "b_0": -3.60112,  # -3.72015 /// -3.60112
                        "b_1": 1.86525e5,  # 6.78215e5 /// 1.86525e5
                        "m_1": -13.5755,  # -14.2532 /// -13.5755
                        "a_3": 1.98119e-3,  # 1.58225e-3 /// 1.98119e-3
                        "m_3": 9.25894,  # 8.71891 /// 9.25894
                    }
                elif component.name == "Graphite-ZDA04":
                    self.n_params.append(12)
                    self.parameters["Graphite-ZDA04"] = {
                        "A": 1.901190e-3 / (5.34e-22),
                        "c_0": -10.1149,
                        "b_0": -5.3308,
                        "b_1": 7.54276e-2,
                        "a_1": 8.08703e-2,
                        "m_1": 3.37644,
                        "b_3": 1.12502e3,
                        "a_3": 0.145378,
                        "m_3": 3.49042,
                        "b_4": 1.12602e3,
                        "a_4": 0.169079,
                        "m_4": 3.636654,
                    }
                elif component.name == "Silicates-ZDA04":
                    self.n_params.append(12)
                    self.parameters["Silicates-ZDA04"] = {
                        "A": 1.541199e-3 / (5.34e-22),
                        "c_0": -8.53081,
                        "b_0": -3.70009,
                        "b_1": 3.96003e-9,
                        "a_1": 9.11246e-3,
                        "m_1": 47.0606,
                        "b_3": 1.48e3,
                        "a_3": 0.484381,
                        "m_3": 12.3253,
                        "b_4": 1.481e3,
                        "a_4": 0.474035,
                        "m_4": 12.0995,
                    }
                elif component.name == "ACH2-ZDA04":
                    self.n_params.append(6)
                    self.parameters["ACH2-ZDA04"] = {
                        "A": 7.862901e-8 / (5.34e-22),
                        "c_0": -3.92513,
                        "b_0": -3.54913,
                        "b_1": 2.13708e-17,
                        "a_1": 2.03908e-4,
                        "m_1": 34.7835,
                    }
                elif component.name == "Silicates1-ZDA04":
                    self.n_params.append(6)
                    self.parameters["Silicates1-ZDA04"] = {
                        "A": 3.680573e-3 / (5.34e-22),
                        "c_0": -8.88283,
                        "b_0": -3.69508,
                        "b_1": 2.17105e-20,
                        "a_1": 3e-7,
                        "m_1": 29.2,
                    }
                elif component.name == "Silicates2-ZDA04":
                    self.n_params.append(9)
                    self.parameters["Silicates2-ZDA04"] = {
                        "A": 6.218762e-9 / (5.34e-22),
                        "c_0": 9.04443e3,
                        "b_0": 5.7679e3,
                        "b_1": 5.77024e3,
                        "a_1": 2.7051e-2,
                        "m_1": 1.00024,
                        "b_2": 3.82848e2,
                        "a_2": 9.39615e-2,
                        "m_2": 8.94494,
                    }
                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )
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

        if len(params) == 9:
            A, c_0, b_0, b_1, a_1, m_1, b_2, a_2, m_2 = params
            # b_3 = a_3 = m_3 = b_4 = a_4 = m_4 = 0
            term3 = b_2 * (np.abs(np.log10(a / a_2)) ** m_2)
            term4 = 0.0
            term5 = 0.0

        elif len(params) == 7:
            A, c_0, b_0, b_1, m_1, a_3, m_3 = params
            a_1 = 1
            b_3 = 1e24
            # b_2 = a_2 = m_2 = b_4 = a_4 = m_4 = 0
            term3 = 0.0
            term4 = b_3 * np.power(np.abs(a - a_3), m_3)
            term5 = 0.0

        elif len(params) == 6:
            A, c_0, b_0, b_1, a_1, m_1 = params
            # b_2 = a_2 = m_2 = b_3 = a_3 = m_3 = b_4 = a_4 = m_4 = 0
            term3 = 0.0
            term4 = 0.0
            term5 = 0.0

        else:
            A, c_0, b_0, b_1, a_1, m_1, b_3, a_3, m_3, b_4, a_4, m_4 = params
            # b_2 = a_2 = m_2 = 0
            term3 = 0.0
            term4 = b_3 * np.power(np.abs(a - a_3), m_3)
            term5 = b_4 * np.power(np.abs(a - a_4), m_4)

        term1 = b_0 * np.log10(a)
        term2 = b_1 * np.power(np.abs(np.log10(a / a_1)), m_1)

        ga = c_0 + term1 - term2 - term3 - term4 - term5

        sizedist = A * (np.float64(10) ** np.float64(ga))

        if composition == "ACH2-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 0.02, a > 0.28))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "PAH-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 5e-3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Silicates-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 0.34))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Graphite-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 0.3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Silicates1-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 3.5e-4, a > 0.024))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "Silicates2-ZDA04":
            (indxs,) = np.where(np.logical_or(a < 0.026, a > 0.37))
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
            if component.name == "PAH-ZDA04":
                self.parameters["PAH-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "m_1": cparams[4],
                    "a_3": cparams[5],
                    "m_3": cparams[6],
                }
            elif component.name == "Graphite-ZDA04":
                self.parameters["Graphite-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "a_1": cparams[4],
                    "m_1": cparams[5],
                    "b_3": cparams[6],
                    "a_3": cparams[7],
                    "m_3": cparams[8],
                    "b_4": cparams[9],
                    "a_4": cparams[10],
                    "m_4": cparams[11],
                }
            elif component.name == "Silicates-ZDA04":
                self.parameters["Silicates-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "a_1": cparams[4],
                    "m_1": cparams[5],
                    "b_3": cparams[6],
                    "a_3": cparams[7],
                    "m_3": cparams[8],
                    "b_4": cparams[9],
                    "a_4": cparams[10],
                    "m_4": cparams[11],
                }
            elif component.name == "ACH2-ZDA04":
                self.parameters["ACH2-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "a_1": cparams[4],
                    "m_1": cparams[5],
                }
            elif component.name == "Silicates1-ZDA04":
                self.parameters["Silicates1-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "a_1": cparams[4],
                    "m_1": cparams[5],
                }
            elif component.name == "Silicates2-ZDA04":
                self.parameters["Silicates2-ZDA04"] = {
                    "A": cparams[0],
                    "c_0": cparams[1],
                    "b_0": cparams[2],
                    "b_1": cparams[3],
                    "a_1": cparams[4],
                    "m_1": cparams[5],
                    "b_2": cparams[6],
                    "a_2": cparams[7],
                    "m_2": cparams[8],
                }
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
            if component.name != "PAH-ZDA04":
                if cparams[4] <= 0.0:
                    lnp_bound = -np.inf

        if not (0.25 <= params[-1] <= 20):
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
            for component in self.components:
                if component.name == "Carbonaceous-HD23":
                    self.n_params.append(2)
                    self.parameters["Carbonaceous-HD23"] = {
                        "B_1": 7.52e-7 / (3.24e-22),
                        "B_2": 8.09e-10 / (3.24e-22),
                    }
                elif component.name == "AstroDust-HD23":
                    self.n_params.append(9)
                    self.parameters["AstroDust-HD23"] = {
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
                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )
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
            B_1, B_2 = params
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

        if composition == "Carbonaceous-HD23":
            (indxs,) = np.where(np.logical_or(a < 4, a > 1e3))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "AstroDust-HD23":
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
            if component.name == "Carbonaceous-HD23":
                self.parameters["Carbonaceous-HD23"] = {
                    "B_1": cparams[0],
                    "B_2": cparams[1],
                }
            elif component.name == "AstroDust-HD23":
                self.parameters["AstroDust-HD23"] = {
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

            if component.name == "Carbonaceous-HD23":
                # keep the normalization always positive
                if cparams[0] < 0.0:
                    lnp_bound = -np.inf
                if cparams[1] < 0.0:
                    lnp_bound = -np.inf

            elif component.name == "AstroDust-HD23":
                if cparams[0] < 0.0:
                    lnp_bound = -np.inf
                if cparams[3] < 0.0:
                    lnp_bound = -np.inf
                if cparams[2] < 0.25:
                    lnp_bound = -np.inf
                if cparams[2] > 0.8:
                    lnp_bound = -np.inf

        if not (0.25 <= params[-1] <= 20):
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
            for component in self.components:
                if component.name == "a-C-Y24":
                    self.n_params.append(5)
                    self.parameters["a-C-Y24"] = {
                        "A": 1.3412712e-18 / (5.34e-22),
                        "alpha": -5,
                        "a_C": 0.05,
                        "a_t": 0.01,
                        "gamma": 1,
                    }
                elif component.name == "a-C:H-Y24":
                    self.n_params.append(3)
                    self.parameters["a-C:H-Y24"] = {
                        "A": 3.1841239e-9 / (5.34e-22),
                        "a_0": 6.195341e-3,
                        "sigma": 1.315171,
                    }
                elif component.name == "aSil-2-Y24":
                    self.n_params.append(3)
                    self.parameters["aSil-2-Y24"] = {
                        "A": 6.9843816e-6 / (5.34e-22),
                        "a_0": 9.210816e-04,
                        "sigma": 1.217290,
                    }
                else:
                    raise ValueError(
                        "%s grain material note supported for this size distribution"
                        % component.name
                    )
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
            A, alpha, a_C, a_t, gamma = params
            sigma = None

        else:
            A, a_0, sigma = params

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

        if composition == "a-C:H-Y24":
            (indxs,) = np.where(np.logical_or(a < 0.04495579, a > 0.7))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "aSil-2-Y24":
            (indxs,) = np.where(np.logical_or(a < 0.011, a > 0.3737511))
            if len(indxs) > 0:
                sizedist[indxs] = 0.0

        if composition == "a-C-Y24":
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
            if component.name == "a-C-Y24":
                self.parameters["a-C-Y24"] = {
                    "A": cparams[0],
                    "alpha": cparams[1],
                    "a_C": cparams[2],
                    "a_t": cparams[3],
                    "gamma": cparams[4],
                }
            elif component.name == "a-C:H-Y24":
                self.parameters["a-C:H-Y24"] = {
                    "A": cparams[0],
                    "a_0": cparams[1],
                    "sigma": cparams[2],
                }
            elif component.name == "aSil-2-Y24":
                self.parameters["aSil-2-Y24"] = {
                    "A": cparams[0],
                    "a_0": cparams[1],
                    "sigma": cparams[2],
                }
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

            if component.name == "a-C-Y24":
                if cparams[2] <= 0:
                    lnp_bound = -np.inf
                if cparams[3] <= 0:
                    lnp_bound = -np.inf

            elif component.name in ["aSil-2-Y24", "a-C:H-Y24"]:
                if cparams[1] <= 0:
                    lnp_bound = -np.inf

        if not (0.25 <= params[-1] <= 20):
            lnp_bound = -np.inf

        if lnp_bound < 0.0:
            return lnp_bound
        else:
            dustmodel.set_size_dist(params)
            return dustmodel.lnprob_generic(obsdata) + lnp_bound
