import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf_grid(y_grid, mu, logvar, training_std):
    var = np.exp(logvar) * training_std**2
    coeff = 1.0 / np.sqrt(2 * np.pi * var)
    exp_term = np.exp(-0.5 * (y_grid - mu) ** 2 / var)
    return coeff * exp_term


def shash_pdf_grid(y_grid, mu, sigma, gamma, tau):
    z = np.arcsinh((y_grid - mu) / sigma)
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    num = tau * phi * np.cosh((z + gamma) * tau)
    den = sigma * np.sqrt(1 + ((y_grid - mu) / sigma) ** 2)
    return num / den

def plot_distribution_heatmap(y_true, mu, logvar=None, sigma=None, gamma=None, tau=None,
                              dist="gaussian", bins=100, training_std=1, margin=0.1):
    y_true = np.array(y_true)
    mu = np.array(mu)

    # Data range
    y_min, y_max = y_true.min(), y_true.max()
    data_range = y_max - y_min

    # Extend range by margin fraction (e.g. 0.1 = 10%)
    y_min_ext = y_min - margin * data_range
    y_max_ext = y_max + margin * data_range

    # Grid for predicted values
    y_grid = np.linspace(y_min_ext, y_max_ext, bins)

    # Initialize heatmap
    heatmap = np.zeros((bins, bins))

    for i, yt in enumerate(y_true):
        if dist == "gaussian":
            pdf_vals = gaussian_pdf_grid(y_grid, mu[i], logvar[i], training_std)
        elif dist == "shash":
            pdf_vals = shash_pdf_grid(y_grid, mu[i], sigma[i], gamma[i], tau[i])
        else:
            raise ValueError("dist must be 'gaussian' or 'shash'")

        pdf_vals = np.asarray(pdf_vals).ravel()
        pdf_vals /= pdf_vals.sum()  # normalize

        # Bin index for true value, now using extended grid
        true_bin = np.digitize(yt, y_grid) - 1
        true_bin = np.clip(true_bin, 0, bins-1)

        # Add PDF into that column
        heatmap[:, true_bin] += pdf_vals

    # Plot with extended extent
    plt.imshow(heatmap, origin="lower", aspect="auto",
               extent=[y_min_ext, y_max_ext, y_min_ext, y_max_ext], cmap="viridis")
    plt.xlabel("True values")
    plt.ylabel("Predicted distribution")
    plt.title(f"{dist.capitalize()} predicted vs true")
    plt.colorbar(label="Density")
    plt.show()
