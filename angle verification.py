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

# Function to select a single galaxy
def select_single_galaxy():
    idx = np.random.randint(0, len(x_positions_filtered))
    x, y, z = x_positions_filtered[idx], y_positions_filtered[idx], z_positions_filtered[idx]
    return np.array([x, y, z])

# Function to generate orthogonal unit vectors
def orthogonal_unit_vectors(g):
    x = np.array([1, 0, 0])
    if np.allclose(g, x):
        x = np.array([0, 1, 0])
    u1 = np.cross(g, x)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(g, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2

# Function to generate the b vector with a given angle theta
def generate_b(g, theta=None, phi=None, sigma=0.5, sigma_fraction = 0.3):
    # Define the desired range for theta
    theta_min = 0.0872665
    theta_max = 0.174533

    if theta is None:
        # Generate theta within the desired range using a Gaussian distribution
        while True:
            theta = np.abs(np.random.normal(loc=0, scale=sigma))  # Take modulus of negative values
            if theta_min <= theta <= theta_max:  # Check if theta is within the desired range
                break

    # Check and print the value of theta for each galaxy
    print(f"Generated theta for this galaxy: {theta} radians")

    if phi is None:
        phi = np.random.uniform(0, 2 * np.pi)  # Generate phi uniformly between 0 and 2π

    # Normalize g
    g_norm = g / np.linalg.norm(g)

    # Generate orthogonal unit vectors
    u1, u2 = orthogonal_unit_vectors(g_norm)

    # Generate b vector
    r = u1 * np.sin(phi) + u2 * np.cos(phi)
    b = g_norm * np.cos(theta) + r * np.sin(theta)
    d_true = np.linalg.norm(g)  # True distance of the galaxy
    d_uncertain = np.random.normal(loc=d_true, scale=sigma_fraction * d_true)  # Add Gaussian uncertainty
    b = b * (d_uncertain / d_true)
    #Scale b by the magnitude of the original g vector
    b = b * np.linalg.norm(g)

    return b


# Function to generate multiple b vectors
def generate_multiple_b_vectors(galaxy_positions, sigma=0.5, sigma_fraction = 0.3):
    perturbed_positions = []
    for g in galaxy_positions:
        b = generate_b(g, sigma=sigma, sigma_fraction = 0.3)
        perturbed_positions.append(b)
    return perturbed_positions

# Function to calculate the angle theta between g and b
def calculate_theta(g, b):
    dot_product = np.dot(g, b)
    magnitude_g = np.linalg.norm(g)
    magnitude_b = np.linalg.norm(b)
    cos_theta = dot_product / (magnitude_g * magnitude_b)
    theta_rad = np.arccos(np.clip(cos_theta, -1, 1))  # Ensure valid range for arccos
    theta_deg = np.degrees(theta_rad)
    return theta_deg

# Function to select multiple galaxies
def select_multiple_galaxies(n_galaxies):
    galaxy_positions = [select_single_galaxy() for _ in range(n_galaxies)]
    return galaxy_positions

# Function to plot multiple g and b vectors
def plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')  # You can choose any colormap (e.g., 'plasma', 'inferno', 'magma')
    normalize = plt.Normalize(vmin=min(d_true_list), vmax=max(d_true_list))  # Normalize distances for colormap


    # Plot all g vectors
    for i, (g, d_true) in enumerate(zip(galaxy_positions, d_true_list)):
        ax.scatter(g[0], g[1], g[2], color='red', marker='o', s=10, label=f'Host Galaxy {i+1} (g)\nTrue Distance: {d_true:.2f} Mpc')

    # Plot all b vectors
    for i, (b, d_gw_samples) in enumerate(zip(perturbed_positions, d_gw_samples_list)):
        d_gw_mean = np.mean(d_gw_samples)  # Mean GW distance
        ax.scatter(b[0], b[1], b[2], color='blue', marker='o', s=10, label=f'Offset Position {i+1} (b)\nMean GW Distance: {d_gw_mean:.2f} Mpc')
    
    # for g, b in zip(galaxy_positions, perturbed_positions):
      
    #   ax.quiver(
    #     g[0], g[1], g[2],  # Start point of the arrow (original galaxy position)
    #     b[0] - g[0], b[1] - g[1], b[2] - g[2],  # Direction of the arrow (vector from g to b)
    #     color='black', arrow_length_ratio=0.3, linewidth=1, alpha = 0.5
    #     )


    # Set equal scaling for the axes
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])

    # Add labels and legend
    ax.set_xlabel('X Position (Mpc)')
    ax.set_ylabel('Y Position (Mpc)')
    ax.set_zlabel('Z Position (Mpc)')
    ax.set_title('3D Plot of Multiple Host Galaxies and Offset Positions')

    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys())

    plt.show()

def plot_offset_kde(galaxy_positions, perturbed_positions):
    # Calculate the offset vectors (b - g)
    offsets = np.array(perturbed_positions) - np.array(galaxy_positions)

    # Create a grid for the KDE
    x = offsets[:, 0]
    y = offsets[:, 1]
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Calculate the KDE for the X-Y plane
    kernel = gaussian_kde(np.vstack([x, y]))
    density = np.reshape(kernel(positions).T, xx.shape)

    # Plot the KDE in the X-Y plane
    plt.figure(figsize=(10, 8))
    plt.imshow(np.rot90(density), cmap='viridis', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(label='Density')
    plt.xlabel('Offset in X (Mpc)')
    plt.ylabel('Offset in Y (Mpc)')
    plt.title('2D KDE Plot of Offsets in the X-Y Plane')
    plt.grid(True)
    plt.show()

# Main script
n_galaxies = 20
galaxy_positions = select_multiple_galaxies(n_galaxies)
perturbed_positions = generate_multiple_b_vectors(galaxy_positions, sigma_fraction=0.3)

# Calculate theta for each pair of g and b
theta_list = [calculate_theta(g, b) for g, b in zip(galaxy_positions, perturbed_positions)]

# Print the results
for i, theta in enumerate(theta_list):
    print(f"Galaxy {i+1}: Theta = {theta:.2f} degrees")

# Calculate true distances and GW distances for each galaxy
d_true_list = [np.linalg.norm(g) for g in galaxy_positions]
d_gw_samples_list = [np.random.normal(loc=d_true, scale=0.3 * d_true, size=1000) for d_true in d_true_list]

# Plot the results
plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list)
