import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import corner
import math


def load_nautilus_checkpoint(fname, n_live):

    dead_samples = []
    dead_logl = []

    with h5py.File(fname, "r") as f:

        sampler = f["sampler"]
        point_keys = sorted(
            [
                k
                for k in sampler.keys()
                if k.startswith("points_") and k not in ("points_0", "points_t")
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        for pk in point_keys:
            idx = pk.split("_")[1]
            lk = f"log_l_{idx}"

            dead_samples.append(np.array(sampler[pk]))
            dead_logl.append(np.array(sampler[lk]))

    dead_samples = np.vstack(dead_samples)
    dead_logl = np.concatenate(dead_logl)

    print("Dead samples:", dead_samples.shape)

    # nested weights
    n = len(dead_logl)
    i = np.arange(1, n + 1)

    X_prev = np.exp(-(i - 1) / n_live)
    X_next = np.exp(-i / n_live)

    weights = X_prev - X_next
    weights /= weights.sum()

    return dead_samples, weights, dead_logl


def posterior_resample(samples, weights, n_draws):
    idx = np.random.choice(len(samples), size=n_draws, replace=False, p=weights)
    return samples[idx]


def corner_in_chunks(
    samples, weights, logl, chunk_size=5, tag="posterior", prior_file="none"
):
    """
    Make multiple weighted corner plots and overlay the
    highest-likelihood point as red lines.
    """

    n_dim = samples.shape[1]
    n_chunks = math.ceil(n_dim / chunk_size)

    # highest likelihood (global)
    idx_max = np.argmax(logl)

    for c in range(n_chunks):

        start = c * chunk_size
        end = min((c + 1) * chunk_size, n_dim)

        uppers, lowers = np.loadtxt(prior_file, unpack=True)
        ratio = uppers / lowers
        samples_phys = lowers * (ratio**samples)

        s_chunk = samples_phys[:, start:end]
        theta_max = samples_phys[idx_max]
        truths_subset = theta_max[start:end]

        labels = [f"p{i}" for i in range(start, end)]  # labels must match exactly

        fig = corner.corner(
            s_chunk,
            weights=weights,
            labels=labels,
            truths=truths_subset,  # <-- red ML marker
            truth_color="red",
            show_titles=True,
            title_fmt=".3g",
            quantiles=[0.16, 0.5, 0.84],
            range=np.repeat(0.9, len(labels)),
        )
        plt.show()
        plt.close(fig)

    return theta_max, idx_max


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("nlive", type=int)
    parser.add_argument("tag")
    parser.add_argument("priors")
    args = parser.parse_args()

    samples, weights, logl = load_nautilus_checkpoint(args.filename, n_live=args.nlive)

    print("N_eff =", 1 / np.sum(weights**2))
    print("Weight sum:", weights.sum())
    print("Dead:", len(samples))
    print("target ~", 5 * args.nlive, "to", 10 * args.nlive)

    theta_max, idx_max = corner_in_chunks(
        samples, weights, logl, chunk_size=5, tag="posterior", prior_file=args.priors
    )

    np.savetxt(
        f"{args.tag}.txt", theta_max, header="Highest likelihood parameters", fmt="%.8f"
    )

    print("Highest likelihood parameters (first 5):", theta_max[:5])
    print("Log-likelihood:", logl[idx_max])


if __name__ == "__main__":
    main()
