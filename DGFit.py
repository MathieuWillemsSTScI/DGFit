#!/usr/bin/env python
#
# DGFit - program to determine dust grain size & composition from fitting dust grain observables
#  dust observables = extinction, scattering properties, depletions, emission, etc.
#
# Started: Jan 2015 (KDG)
#  added IR emission: Feb 2015 (KDG)
#  udpated to move plotting to a separate function: Dec 2015 (KDG)
# 
from __future__ import print_function

import math
import time
import argparse

import numpy as np
import matplotlib.pyplot as pyplot
from astropy.io import fits

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import emcee
import triangle

import DustModel
import ObsData
#import ObsData_Azv18 as ObsData

# compute the ln(prob) for an input set of model parameters
def lnprobsed(params, ObsData, DustModel):

    # make sure the size distributions are all positve
    for param in params:
        if param < 0.0:
            return -np.inf

    # update the size distributions
    #  the input params are the concatenated size distributions
    DustModel.set_size_dist(params)

    # get the integrated dust properties
    cabs, csca, natoms, emission = DustModel.eff_grain_props()
    #print(natoms)
    cext = cabs + csca
    dust_alnhi = 1.086*cext
    
    # compute the ln(prob) for A(l)/N(HI)
    lnp_alnhi = -0.5*np.sum((((obsdata.alnhi - dust_alnhi)/(0.10*obsdata.alnhi))**2))
    #lnp_alnhi /= obsdata.n_wavelengths

    # compute the ln(prob) for the depletions
    lnp_dep = 0.0
    if obsdata.fit_depletions:
        for atomname in natoms.keys():
            if natoms[atomname] > 1.5*obsdata.total_abundance[atomname][0]: # hard limit at 1.5x the total possible abundaces (all atoms in dust)
                #print('boundary issue')
                return -np.inf
            elif natoms[atomname] > obsdata.depletions[atomname][0]: # only add if natoms > depletions
                lnp_dep = ((natoms[atomname] - obsdata.depletions[atomname][0])/obsdata.depletions[atomname][1])**2
        lnp_dep *= -0.5

    # compute the ln(prob) for IR emission
    lnp_emission = 0.0
    if obsdata.fit_ir_emission:
        lnp_emission = -0.5*np.sum((((obsdata.ir_emission - emission[obsdata.ir_emission_indxs])/(obsdata.ir_emission_uncs))**2))

    # compute the ln(prob) for the dust albedo
    lnp_albedo = 0.0
    if obsdata.fit_scat_param:
        albedo = csca/cext
        lnp_albedo = -0.5*np.sum((((obsdata.scat_albedo - albedo[obsdata.scat_indxs])/(obsdata.scat_albedo_unc))**2))

    # combine the lnps
    lnp = lnp_alnhi + lnp_dep + lnp_emission + lnp_albedo
    
    if math.isinf(lnp) | math.isnan(lnp):
        print(lnp)
        print(params)
        exit()
    else:
        return lnp

# main fitting code
if __name__ == "__main__":
    
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fast", help="Use minimal walkers, n_steps, n_burn to debug code",
                        action="store_true")
    parser.add_argument("-s", "--slow", help="Use lots of walkers, n_steps, n_burn",
                        action="store_true")
    parser.add_argument("-r", "--read", default="",
                        help="Read size distribution from disk")
    parser.add_argument("-t", "--tag", default='dgfit_test',
                        help="basename to use for output files")
    args = parser.parse_args()

    # set the basename of the output
    basename = args.tag

    # save the start time 
    start_time = time.clock()

    # get the dust model 
    min_wave = 0.09
    max_wave = 3.0
    min_wave_emission = 1.0
    max_wave_emission = 1000.0
    dustmodel = DustModel.DustModel(['astro-silicates','astro-carbonaceous'], path='/home/kgordon/Dirty_v2/write_grain/indiv_grain/',
                                    min_wave=min_wave,max_wave=max_wave,
                                    min_wave_emission=min_wave_emission,max_wave_emission=max_wave_emission)

    # get the observed data to fit
    obsdata = ObsData.ObsData(dustmodel.components[0].wavelengths, dustmodel.components[0].wavelengths_emission)

    # replace the default size distribution with one from a file
    if args.read != "":
        for k, component in enumerate(dustmodel.components):
            fitsdata = fits.getdata(args.read,k+1)
            if len(component.size_dist) != len(fitsdata[:][1]):
                component.size_dist = 10.**np.interp(np.log10(component.sizes),np.log10(fitsdata['SIZE']), np.log10(fitsdata['DIST']))
            else:
                component.size_dist = fitsdata['DIST']
                #if component.sizes[i] > 0.5e-4:
                #    component.size_dist[i] *= 1e-4
    else:
        # check that the default size distributions give approximately the right level of the A(lambda)/N(HI) curve
        #  if not, adjust the overall level of the size distributions to get them close
        cabs, csca, natoms, emission = dustmodel.eff_grain_props()
        dust_alnhi = 1.086*(cabs + csca)
        ave_model = np.average(dust_alnhi)
        ave_data = np.average(obsdata.alnhi)
        ave_ratio = ave_data/ave_model
        if (ave_ratio < 0.5) | (ave_ratio > 2):
            for component in dustmodel.components:
                component.size_dist *= ave_ratio
            
        cabs, csca, natoms, emission = dustmodel.eff_grain_props()
        max_violation = 0.0
        for atomname in natoms.keys():
            cur_violation = natoms[atomname]/obsdata.depletions[atomname][0]
            if cur_violation > max_violation:
                max_violation = cur_violation

        #if max_violation > 2:
        #    for component in dustmodel.components:
        #        component.size_dist *= 1.9/max_violation        

    # save the starting model
    dustmodel.save(basename + '_sizedist_start.fits')
    
    # setup time
    setup_time = time.clock()
    print('setup time taken: ',(setup_time - start_time)/60., ' min')

    # inital guesses at parameters
    p0 = dustmodel.components[0].size_dist
    for k in range(1,dustmodel.n_components):
        p0 = np.concatenate([p0,dustmodel.components[k].size_dist])

    # call scipy.optimize to get a better initial guess
    #print(p0)
    #better_start = minimize(lnprobsed, p0, args=(obsdata, dustmodel))
    #print(better_start)
    #exit()

    #import scipy.optimize as op
    #nll = lambda *args: -lnprobsed(*args)
    #result = op.minimize(nll, p0, args=(obsdata, dustmodel))
    #print(result)
    #exit()

    # trying with lmfit (not tested)
    #params = Parameters()
    #for i in range(n_params):
    #   params.add('p'+str(i),value=p0[i],min=0.0)
    #out = minimize(kext_residuals, params, args=(1.0/xdata, ydata))
    #print(out.params)

    ndim = len(p0)
    print('# params = ', ndim)
    if args.fast:
        print('using the fast params')
        nwalkers = 2*ndim
        nsteps = 50
        burn   = 5
    elif args.slow:
        print('using the slow params')
        nwalkers = 2*ndim
        nsteps = 5000
        burn   = 20000
    else:
        nwalkers = 2*ndim
        nsteps = 1000
        burn   = 5000

    # setting up the walkers to start "near" the inital guess
    p  = [ p0*(1+0.25*np.random.normal(0,1.,ndim))  for k in range(nwalkers)]

    # ensure that all the walkers start with positive values
    for pc in p:
        for pcs in pc:
            if pcs <= 0.0:
                pcs = 0.0

    # make sure each walker starts with allowed abundances
    for pc in p:
        dustmodel.set_size_dist(pc)
        cabs, csca, natoms, emission = dustmodel.eff_grain_props()
        max_violation = 0.0
        for atomname in natoms.keys():
            cur_violation = natoms[atomname]/(obsdata.depletions[atomname][0] + obsdata.depletions[atomname][1])
            if cur_violation > max_violation:
                max_violation = cur_violation
        if max_violation > 2:
            pc *= 1.9/max_violation        
            
    # setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobsed, args=(obsdata, dustmodel), threads=4)

    # burn in the walkers
    pos, prob, state = sampler.run_mcmc(p, burn)

    # rest the sampler
    sampler.reset()

    # do the full sampling
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)

    # untested code from emcee webpages for incrementally saving the chains
    #f = open("chain.dat", "w")
    #f.close()
    #for k, result in enumerate(sampler.sample(pos0, iterations=500, storechain=False)):
    #    print(k)
    #    position = result[0]
    #    f = open("chain.dat", "a")
    #    for k in range(position.shape[0]):
    #        f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
    #    f.close()

    emcee_time = time.clock()
    print('emcee time taken: ',(emcee_time - setup_time)/60., ' min')

    # get the best fit values
    max_lnp = -1e6
    for k in range(nwalkers):
        tmax_lnp = np.max(sampler.lnprobability[k])
        if tmax_lnp > max_lnp:
            max_lnp = tmax_lnp
            indxs, = np.where(sampler.lnprobability[k] == tmax_lnp)
            fit_params_best = sampler.chain[k,indxs[0],:]

    dustmodel.set_size_dist(fit_params_best)
    cabs_best, csca_best, natoms_best, emission_best = dustmodel.eff_grain_props()

    # save the best fit size distributions
    dustmodel.save(basename + '_sizedist_best.fits')

    # get the 50p values and uncertainties
    samples = sampler.chain.reshape((-1, ndim))
    values = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.percentile(samples, [16, 50, 84],
                                    axis=0)))
    fin_size_dist_50p, fin_size_dist_punc, fin_size_dist_munc = zip(*values)

    # 50p dust params
    dustmodel.set_size_dist(fin_size_dist_50p)
    cabs, csca, natoms, emission = dustmodel.eff_grain_props()
    dust_alnhi = 1.086*(cabs + csca)

    # save the final size distributions
    dustmodel.save(basename + '_sizedist.fits', size_dist_uncs=[fin_size_dist_punc, fin_size_dist_munc])
    
