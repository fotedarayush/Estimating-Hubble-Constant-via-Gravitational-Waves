import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import gaussian_kde

# Load data
file_path = "/Users/ayushfotedar/Documents/eag_ss_combined.csv"
data = pd.read_csv(file_path)

# Extract relevant columns
galaxy_ids = data.iloc[:, 0].to_numpy()
x_positions = data.iloc[:, 5].to_numpy()
y_positions = data.iloc[:, 6].to_numpy()
z_positions = data.iloc[:, 7].to_numpy()
stellar_mass = data["M_star_tot [M_sol]"].to_numpy()

# Apply mass threshold
mass_threshold = 1e6
mask = stellar_mass >= mass_threshold
x_positions_filtered = x_positions[mask]
y_positions_filtered = y_positions[mask]
z_positions_filtered = z_positions[mask]

def select_single_galaxy():
    idx = np.random.randint(0, len(x_positions_filtered))
    x, y, z = x_positions_filtered[idx], y_positions_filtered[idx], z_positions_filtered[idx]
    return np.array([x, y, z])

def orthogonal_unit_vectors(g):
    x = np.array([1, 0, 0])
    if np.allclose(g, x):
        x = np.array([0, 1, 0])
    u1 = np.cross(g, x)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(g, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2

# Fisher sampling function
def sample_gamma_fisher_truncated(sigma_tot=5, gamma_max=np.radians(30)):
    k = 1.0 / (0.66 * sigma_tot**2)
    F_max = (np.exp(k) - np.exp(k * np.cos(gamma_max))) / (np.exp(k) - np.exp(-k))
    u = np.random.uniform(0, 1)
    u_trunc = u * F_max
    A = np.exp(k) - np.exp(-k)
    val = np.exp(k) - u_trunc * A
    cos_gamma = np.log(val) / k
    cos_gamma = np.clip(cos_gamma, -1, 1)
    gamma = np.arccos(cos_gamma)
    return gamma, k

# Probability between gamma1 and gamma2
def cumulative_probability(gamma1, gamma2, k):
    return (np.exp(k * np.cos(gamma1)) - np.exp(k * np.cos(gamma2))) / (np.exp(k) - np.exp(-k))

# Example usage
example_k = 1.0 / (0.66 * 5**2)
prob_example = cumulative_probability(np.radians(5), np.radians(20), example_k)
print(f"Probability between 5° and 20°: {prob_example:.4f}")

# Generate b
def generate_b(g, sigma_tot=5, sigma_fraction=0.3, gamma_max=np.radians(30)):
    g_norm = g / np.linalg.norm(g)
    u1, u2 = orthogonal_unit_vectors(g_norm)
    
    gamma, k = sample_gamma_fisher_truncated(sigma_tot=sigma_tot, gamma_max=gamma_max)
    print(f"Sampled gamma (offset angle): {np.degrees(gamma):.2f} degrees (k={k:.4f})")
    
    phi = np.random.uniform(0, 2 * np.pi)
    r = u1 * np.sin(phi) + u2 * np.cos(phi)
    b_direction = g_norm * np.cos(gamma) + r * np.sin(gamma)
    
    d_true = np.linalg.norm(g)
    d_uncertain = np.random.normal(loc=d_true, scale=sigma_fraction * d_true)
    b = b_direction * d_uncertain
    return b, gamma

def generate_multiple_b_vectors(galaxy_positions, sigma_tot=5, sigma_fraction=0.3):
    perturbed_positions = []
    gamma_list = []
    for g in galaxy_positions:
        b, gamma = generate_b(g, sigma_tot=sigma_tot, sigma_fraction=sigma_fraction)
        perturbed_positions.append(b)
        gamma_list.append(gamma)
    return perturbed_positions, gamma_list

def calculate_theta(g, b):
    dot_product = np.dot(g, b)
    magnitude_g = np.linalg.norm(g)
    magnitude_b = np.linalg.norm(b)
    cos_theta = dot_product / (magnitude_g * magnitude_b)
    theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def select_multiple_galaxies(n_galaxies):
    return [select_single_galaxy() for _ in range(n_galaxies)]

def plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (g, d_true) in enumerate(zip(galaxy_positions, d_true_list)):
        ax.scatter(g[0], g[1], g[2], color='red', marker='o', s=10, label=f'Host Galaxy {i+1}' if i==0 else "")
    for i, (b, d_gw_samples) in enumerate(zip(perturbed_positions, d_gw_samples_list)):
        ax.scatter(b[0], b[1], b[2], color='blue', marker='o', s=10, label=f'Offset {i+1}' if i==0 else "")
    
    for g, b in zip(galaxy_positions, perturbed_positions):
      ax.quiver(
        g[0], g[1], g[2],
        b[0] - g[0], b[1] - g[1], b[2] - g[2],
        color='black', arrow_length_ratio=0.3, linewidth=1, alpha=0.5
      )
    
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    ax.set_xlabel('X Position (Mpc)')
    ax.set_ylabel('Y Position (Mpc)')
    ax.set_zlabel('Z Position (Mpc)')
    ax.set_title('3D Plot of Host Galaxies and Offset Positions')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()

def plot_offset_kde(galaxy_positions, perturbed_positions):
    offsets = np.array(perturbed_positions) - np.array(galaxy_positions)
    x = offsets[:, 0]
    y = offsets[:, 1]
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(np.vstack([x, y]))
    density = np.reshape(kernel(positions).T, xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.rot90(density), cmap='viridis', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(label='Density')
    plt.xlabel('Offset in X (Mpc)')
    plt.ylabel('Offset in Y (Mpc)')
    plt.title('2D KDE Plot of Offsets in the X-Y Plane')
    plt.grid(True)
    plt.show()

# ------------- MAIN SCRIPT -------------
n_galaxies = 20
galaxy_positions = select_multiple_galaxies(n_galaxies)
perturbed_positions, gamma_list = generate_multiple_b_vectors(galaxy_positions, sigma_tot=5, sigma_fraction=0.3)

theta_list = [calculate_theta(g, b) for g, b in zip(galaxy_positions, perturbed_positions)]
for i, theta in enumerate(theta_list):
    print(f"Galaxy {i+1}: Angle between g and b = {theta:.2f} degrees")

d_true_list = [np.linalg.norm(g) for g in galaxy_positions]
d_gw_samples_list = [np.random.normal(loc=d_true, scale=0.3 * d_true, size=1000) for d_true in d_true_list]

plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list)

# Optionally plot KDE
# plot_offset_kde(galaxy_positions, perturbed_positions)

# --- New Code Here: Theoretical PDF/CDF Overlay ---
# 1) Convert gamma_list to degrees
gamma_degrees = np.degrees(gamma_list)

# 2) Plot histogram (PDF) + cumulative histogram (CDF)
plt.figure(figsize=(10, 6))
counts_pdf, bins_pdf, _ = plt.hist(gamma_degrees, bins=20, density=True, alpha=0.6, label='PDF of $\gamma$')
counts_cdf, bins_cdf, _ = plt.hist(gamma_degrees, bins=20, density=True, cumulative=True,
                                   histtype='step', linewidth=2, label='CDF of $\gamma$')

plt.xlabel('Offset angle $\gamma$ (degrees)')
plt.ylabel('Probability')
plt.title('Histogram and Cumulative Distribution of Offset Angles (Fisher Sampling)')

# 3) Define truncated Fisher PDF & CDF

def fisher_pdf_truncated(gamma_rad, k, gamma_max=np.radians(30)):
    """Compute truncated Fisher PDF (numerically normalized) on [0..gamma_max]."""
    denom = np.exp(k) - np.exp(-k)
    p_raw = (k/denom) * np.sin(gamma_rad) * np.exp(k*np.cos(gamma_rad))
    # Numerically find total area from 0..gamma_max
    g_grid = np.linspace(0, gamma_max, 300)
    raw_vals = (k/denom) * np.sin(g_grid) * np.exp(k*np.cos(g_grid))
    area_raw = np.trapz(raw_vals, g_grid)
    return p_raw / area_raw

def fisher_cdf_truncated(gamma_rad, k, gamma_max=np.radians(30)):
    """Compute truncated Fisher CDF on [0..gamma_max]."""
    # raw CDF
    denom = np.exp(k) - np.exp(-k)
    F_raw = (np.exp(k) - np.exp(k*np.cos(gamma_rad))) / denom
    # raw CDF at gamma_max
    F_raw_gmax = (np.exp(k) - np.exp(k*np.cos(gamma_max))) / denom
    return F_raw / F_raw_gmax

# 4) Evaluate the theoretical PDF & CDF on a fine grid
gamma_grid_deg = np.linspace(0, 30, 200)
gamma_grid_rad = np.radians(gamma_grid_deg)

# Use the same k as in sample_gamma_fisher_truncated (sigma_tot=5)
k_theory = 1.0 / (0.66 * 5**2)

pdf_vals = fisher_pdf_truncated(gamma_grid_rad, k_theory, np.radians(30))
cdf_vals = fisher_cdf_truncated(gamma_grid_rad, k_theory, np.radians(30))

# 5) Overlay the theoretical PDF & CDF
plt.plot(gamma_grid_deg, pdf_vals, 'r-', label='Theoretical PDF')
plt.plot(gamma_grid_deg, cdf_vals, 'g-', label='Theoretical CDF')

plt.legend()
plt.show()
