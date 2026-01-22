import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import corner

from dgfit.dustmodel import (
    DustModel,
    MRN77DustModel,
    WD01DustModel,
    ZDA04DustModel,
    HD23DustModel,
    Y24DustModel,
)
from dgfit.obsdata import ObsData


def load_nautilus_checkpoint(fname, n_live):

    dead_samples = []
    dead_logl = []

    with h5py.File(fname, "r") as f:

        sampler = f["sampler"]

        # find all chunk numbers automatically
        point_keys = sorted(
            [k for k in sampler.keys() if k.startswith("points_") and k not in ("points_0", "points_t")],
            key=lambda x: int(x.split("_")[1])
        )

        for pk in point_keys:
            idx = pk.split("_")[1]
            lk = f"log_l_{idx}"

            dead_samples.append(np.array(sampler[pk]))
            dead_logl.append(np.array(sampler[lk]))

    dead_samples = np.vstack(dead_samples)
    dead_logl = np.concatenate(dead_logl)

    print("Dead samples:", dead_samples.shape)
    print("Dead logL:", dead_logl.shape)

    # nested weights
    n = len(dead_logl)
    i = np.arange(1, n+1)

    X_prev = np.exp(-(i-1)/n_live)
    X_next = np.exp(-i/n_live)

    weights = X_prev - X_next
    weights /= weights.sum()

    return dead_samples, weights, dead_logl



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("nlive", type=int)
    parser.add_argument(
        "obsfile", help="Data file giving the observational data to be fit"
    )
    parser.add_argument("tag")

    args = parser.parse_args()

    samples, weights, logl = load_nautilus_checkpoint(
        args.filename,
        n_live=args.nlive
    )

    print("N_eff =", 1/np.sum(weights**2))
    print("Weight sum:", weights.sum())
    print("Dead:", len(samples))
    print("target ~", 5*args.nlive, "to", 10*args.nlive)


    # # optional: thin for memory if too large
    # thin_factor = max(1, len(samples)//50000)  # keep ~50k points max
    # samples_thin = samples[::thin_factor]
    # weights_thin = weights[::thin_factor]

    # # make corner plot
    # figure = corner.corner(
    #     samples_thin,
    #     weights=weights_thin,
    #     show_titles=True,
    #     title_fmt=".3f",
    #     labels=[f"param_{i}" for i in range(samples.shape[1])],
    # )
    # plt.show()

    # index of maximum logL
    idx_max = np.argmax(logl)

    # parameters corresponding to maximum log-likelihood
    theta_max = samples[idx_max]

    np.savetxt(f"{args.tag}.txt", theta_max, header="Highest likelihood parameters", fmt="%.8f")

    print("Highest likelihood parameters (first 5):", theta_max[:5])
    print("Log-likelihood:", logl[idx_max])


if __name__ == "__main__":
    main()
