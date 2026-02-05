import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py

def weighted_correlation_matrix(points, log_w):
    """
    points : (Nsamples, Nparams)
    log_w  : log weights

    returns:
        corr : (Nparams, Nparams) correlation matrix
    """
    w = np.exp(log_w)
    w /= np.sum(w)

    # weighted mean
    mean = np.average(points, axis=0, weights=w)
    diff = points - mean

    # weighted covariance matrix
    cov = (w[:, None] * diff).T @ diff
    std = np.sqrt(np.diag(cov))

    # correlation matrix
    corr = cov / np.outer(std, std)

    return corr

def plot_correlation_matrix(corr, labels, basename=None):
    N = len(labels)

    # mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    corr_masked = np.ma.masked_where(mask, corr)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        corr_masked,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    # draw grid lines between cells
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

    ax.tick_params(which="minor", bottom=False, left=False)

    # ticks only for lower triangle look
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))

    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # remove grid above diagonal visually
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient")

    ax.set_title("Parameter Correlations")

    plt.tight_layout()

    if basename:
        fig.savefig(f"{basename}_correlation_triangle.png", dpi=200)

    plt.show()

def correlated_params(corr, labels, threshold=0.5):
    """
    Returns a list of parameter names where |corr| > threshold with all others.
    """
    # make sure labels are plain Python strings
    labels = [str(l) for l in labels]

    N = len(labels)
    pairs = []

    for i in range(N):
        for j in range(i + 1, N):
            if abs(corr[i, j]) > threshold:
                pairs.append((labels[i], labels[j], corr[i, j]))

    return pairs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--tag")
    args = parser.parse_args()

    with h5py.File(f"{args.filename}") as f:
        points = f["points"][:]
        log_w = f["log_w"][:]
        labels = f["labels"][:].astype(str)

    corr = weighted_correlation_matrix(points, log_w)
    plot_correlation_matrix(corr, labels, basename=args.tag)
    corr_params = correlated_params(corr, labels, threshold=0.5)
    for p1, p2, val in corr_params:
        print(f"{p1} vs {p2} -> {val:.2f}")
