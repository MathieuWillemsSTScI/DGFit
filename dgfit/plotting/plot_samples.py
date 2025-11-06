from __future__ import print_function

import argparse
import numpy as np
import corner
import h5py  # for HDF5 support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help=("file with EMCEE sampler chain (.h5)"))
    args = parser.parse_args()

    # open the .h5 file
    with h5py.File(args.filename, "r") as f:
        samples_data = f["mcmc"]["chain"][:]
        nsteps, nwalkers, nparams = samples_data.shape

    samples = samples_data.reshape((-1, nparams))

    print(samples.shape)

    fig = corner.corner(samples)
    fig.savefig("%s.png" % args.filename)


if __name__ == "__main__":
    main()