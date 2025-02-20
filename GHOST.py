from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import os
from colorama import Fore, Back, Style
from scipy.optimize import curve_fit
from playsound import playsound
from scipy.ndimage import gaussian_filter


file_path = "/Users/ayushfotedar/Documents/eag_ss_combined.csv"


data = pd.read_csv(file_path)



galaxy_ids = data.iloc[:, 0].to_numpy()
x_positions = data.iloc[:, 5].to_numpy()  # X positions (3rd column)
y_positions = data.iloc[:, 6].to_numpy()  # Y positions (4th column)
z_positions = data.iloc[:, 7].to_numpy()  # Z positions (5th column)
stellar_mass = data["M_star_tot [M_sol]"].to_numpy()
dm_mass = data["M_DM_tot [M_sol]"].to_numpy()
gas_mass = data["M_Gas_tot [M_sol]"].to_numpy()

mass_threshold = 1e6
mask = stellar_mass >= mass_threshold
x_positions_filtered = x_positions[mask]
y_positions_filtered = y_positions[mask]
z_positions_filtered = z_positions[mask]

BOOT_SOUND = "/Users/ayushfotedar/Documents/year 4 lecture slides/PA4900 Research Project Material/sound files/soft startup.mp3"
OPTION_SOUND = '/Users/ayushfotedar/Documents/year 4 lecture slides/PA4900 Research Project Material/sound files/interface-204503.mp3'
INVALID_SOUND = "invalid_sound.wav"
ERROR_SOUND = '/Users/ayushfotedar/Documents/year 4 lecture slides/PA4900 Research Project Material/sound files/error-call-to-attention-129258.mp3'
EXIT_SOUND = '/Users/ayushfotedar/Documents/year 4 lecture slides/PA4900 Research Project Material/sound files/exit_sound.wav'

output_directory = '/Users/ayushfotedar/Documents/year 4 lecture slides/PA4900 Research Project Material/data files'



colors = np.random.rand(len(x_positions))



def handle_error(error):
    """Handle errors by displaying a message and playing an error sound."""
    print(f"Error: {error}")
    play_sound(ERROR_SOUND)
    

def play_sound(sound_file):
    """Play a sound file."""
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")


def loading_screen():
    print('\nLoading GHOST', end = '')
    
    for i in range(11):
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(0.2)
    
def animated_menu(first_run=True):

    options = [
        '1. 3D Galaxy Distribution Visualiser',
        '2. Sliced Distribution with Contour',
        '3. Galaxy Mass Distribution',
        '4. Random Galaxy Sampling',
        '5. Perform Error Box Simulation (1$\sigma$)',
        '6. Galaxy and its offset position ',
        '7. Multiple galaxies with offset',
        '8. Galaxies and their offset contour slice',
        '9. estimate the Hubble constant',
        '10. Exit'
    ]
    

    if first_run:
        print(Fore.GREEN + '\n\nWelcome to the Galaxy Hubble cOnstant eSTimator (GHOST)\n\n')
        time.sleep(0.2)
        for i in options:
            for j in i:
                sys.stdout.write(j)
                sys.stdout.flush()
                time.sleep(0.02)
            print()
            time.sleep(0.15)
        print('\n\nSelect an option to continue')
    else:
        print(Fore.GREEN + '\n\nGalaxy Hubble cOnstant eSTimator (GHOST)\n\n')
        for option in options:  # Instant display of options
            print(option)
        print('\nSelect an option to continue')

def alt_animated_menu(first_run=True):
    options = [
        '1. 3D Galaxy Distribution Visualiser',
        '2. Sliced Distribution with Contour',
        '3. Galaxy Mass Distribution',
        '4. Random Galaxy Sampling',
        '5. Exit'
    ]
    
    if first_run:
        print(Fore.GREEN + '\nWelcome to the Galaxy Hubble cOnstant eSTimator (GHOST)\n')
        time.sleep(0.3)  # Slight delay for the initial menu
        for option in options:
            print(option)
            time.sleep(0.2)  # Slight delay for each option on the first run
        print('\nSelect an option to continue')
    else:
        print(Fore.GREEN + '\nGalaxy Hubble cOnstant eSTimator (GHOST)\n')
        for option in options:  # Instant display of options
            print(option)
        print('\nSelect an option to continue')


def select_single_galaxy():
    idx = np.random.randint(0, len(x_positions))
    x, y, z = x_positions_filtered[idx], y_positions_filtered[idx], z_positions_filtered[idx]
    return idx, x, y, z





def orthogonal_unit_vectors(g, box_size=1):
    x = np.array([box_size, 0, 0])
    if np.allclose(g, x):  
        x = np.array([0, box_size, 0])
    u1 = np.cross(g, x)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(g, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2

def generate_b(g, theta=None, phi=None, sigma=0.5):
    # Define the desired range for theta
    theta_min = 0.0872665 # approx. 5 degrees
    theta_max = 0.174533 # approx. 10 degrees

    if theta is None:
        # Generate theta within the desired range using a Gaussian distribution
        while True:
            theta = np.abs(np.random.normal(loc=0, scale=sigma))  # Take modulus of negative values
            if theta_min <= theta <= theta_max:  # Check if theta is within the desired range
                break

    # Check and print the value of theta for each galaxy
    print(f"Generated theta for this galaxy: {theta} radians")

    if phi is None:
        phi = np.random.uniform(0, 2 * np.pi)  

    # Normalize g
    g_norm = g / np.linalg.norm(g)

    # Generate orthogonal unit vectors
    u1, u2 = orthogonal_unit_vectors(g_norm)

    # Generate b vector
    r = u1 * np.sin(phi) + u2 * np.cos(phi)
    b = g_norm * np.cos(theta) + r * np.sin(theta)

    # Scale b by the magnitude of the original g vector
    b = b * np.linalg.norm(g)

    return b

def generate_multiple_b_vectors(galaxy_positions, sigma=0.5, min_theta_deg=4, max_theta_deg=6):
    perturbed_positions = []
    for g in galaxy_positions:
        b = generate_b(g, sigma=sigma)
        perturbed_positions.append(b)
    return perturbed_positions



def select_multiple_galaxies(n_galaxies):
    galaxy_positions = [select_single_galaxy() for _ in range(n_galaxies)]
    return galaxy_positions


def plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all g vectors
    for i, (g, d_true) in enumerate(zip(galaxy_positions, d_true_list)):
        ax.scatter(g[0], g[1], g[2], color='red', marker='o', s=10, label=f'Host Galaxy {i+1} (g)\nTrue Distance: {d_true:.2f} Mpc')

    # Plot all b vectors
    for i, (b, d_gw_samples) in enumerate(zip(perturbed_positions, d_gw_samples_list)):
        d_gw_mean = np.mean(d_gw_samples)  # Mean GW distance
        ax.scatter(b[0], b[1], b[2], color='blue', marker='o', s=10, label=f'Offset Position {i+1} (b)\nMean GW Distance: {d_gw_mean:.2f} Mpc')
    
    for g, b in zip(galaxy_positions, perturbed_positions):
      
      ax.quiver(
        g[0], g[1], g[2],  # Start point of the arrow (original galaxy position)
        b[0] - g[0], b[1] - g[1], b[2] - g[2],  # Direction of the arrow (vector from g to b)
        color='green', arrow_length_ratio=0.1, linewidth=1
        )


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

# Step 2: Calculate true distance and GW distance
def calculate_distances(x, y, z, sigma_fraction=0.3, n_samples=1000):
    d_true = np.sqrt(x**2 + y**2 + z**2)
    sigma = sigma_fraction * d_true
    d_gw_samples = np.random.normal(loc=d_true, scale=sigma, size=n_samples)
    return d_true, d_gw_samples

# Step 3: Plot histograms
def plot_histograms(d_gw_samples, d_true, idx):
    # Histogram of d_GW / d_true
    ratios = d_gw_samples / d_true
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='True Ratio (1)')
    plt.title(f"Histogram of d_GW / d_true for Galaxy {idx}")
    plt.xlabel("d_GW / d_true")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()

    # Histogram of stellar mass, dark matter mass, and gas mass
    plt.figure(figsize=(10, 6))
    plt.hist(stellar_mass, bins=50, alpha=0.7, label="Stellar Mass", color='blue', log=True)
    plt.hist(dm_mass, bins=50, alpha=0.7, label="Dark Matter Mass", color='green', log=True)
    plt.hist(gas_mass, bins=50, alpha=0.7, label="Gas Mass", color='orange', log=True)
    plt.title("Histogram of Galaxy Masses")
    plt.xlabel("Mass (M_sol)")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plt.grid()
    plt.show()

    # Histogram of all galaxy distances
    all_distances = np.sqrt(x_positions**2 + y_positions**2 + z_positions**2)
    plt.figure(figsize=(8, 6))
    plt.hist(all_distances, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.axvline(d_true, color='red', linestyle='--', label=f"Selected Galaxy Distance: {d_true:.2f}")
    plt.title("Histogram of All Galaxy Distances")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()

# Main function to execute the task
def main():
    print("Selecting a single galaxy...")
    idx, x, y, z = select_single_galaxy()
    print(f"Selected Galaxy Index: {idx}, Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    
    print("Calculating distances...")
    d_true, d_gw_samples = calculate_distances(x, y, z)
    print(f"True Distance (d_true): {d_true:.2f} Mpc")
    
    print("Generating histograms...")
    plot_histograms(d_gw_samples, d_true, idx)
    print("Process Complete.")


def selected_random_galaxy():
    idx = np.random.randint(0, len(x_positions))  # Random index
    return x_positions[idx], y_positions[idx], z_positions[idx]


def cartesian_to_polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # Polar angle
    phi = np.arctan2(y, x)    # Azimuthal angle
    return r, theta, phi

def simulate_error_box(n_samples):
    # Step 1: Select a random galaxy
    x, y, z = selected_random_galaxy()
    print(f"Selected Galaxy Position: x={x}, y={y}, z={z}")
    
    # Step 2: Compute the true distance
    d_true = np.sqrt(x**2 + y**2 + z**2)
    print(f"True Distance (d_true): {d_true:.2f}")
    
    # Step 3: Generate Gaussian distances (GW distances)
    sigma = 0.3 * d_true  # 30% error (1σ)
    d_gw_samples = generate_gaussian_distances(d_true, sigma, n_samples) 
    
    # Step 4: Compute d_gw / d_true for each sample
    ratios = d_gw_samples / d_true
    
    # Step 5: Plot the results
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='True Ratio (1)')
    plt.title("Distribution of d_GW / d_true")
    plt.xlabel("d_GW / d_true")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Step 6: Plot Gaussian in polar coordinates
    r, theta, phi = cartesian_to_polar(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        d_gw_samples * np.sin(theta) * np.cos(phi),
        d_gw_samples * np.sin(theta) * np.sin(phi),
        d_gw_samples * np.cos(theta),
        c=d_gw_samples, cmap='viridis', s=10, alpha=0.5
    )
    ax.set_title("Gaussian Error in Polar Coordinates")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    print("Simulation Complete.")


def ending_screen():
    exit_text = '\nExiting the program. Goodbye!'
    for i in exit_text:
        sys.stdout.write(i)
        sys.stdout.flush()
        time.sleep(0.03)

def Galaxy_Cluster():

    
    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(x_positions, y_positions, z_positions, c=colors, cmap='viridis', s=0.05, alpha=0.6)
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label('Color Legend', fontsize=12)
    #ax.set_title('3D Spatial Plot of Galaxy Positions (Colored)', fontsize=14)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Z Position', fontsize=12)
    plt.show()

def Galaxy_Contour(z_min, z_max, bins=1000):



    slice_mask = (z_positions >= z_min) & (z_positions <= z_max)
    x_slice = x_positions[slice_mask]
    y_slice = y_positions[slice_mask]


    
    hist, xedges, yedges = np.histogram2d(x_slice, y_slice, bins=1000)
    # hist_smoothed = gaussian_filter(hist.T, sigma=sigma)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_slice, y_slice, s=5, alpha=0.7)
    plt.title(f'2D Slice of 3D Data (Z in [{z_min}, {z_max}])', fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(True)
    plt.show()



    plt.figure(figsize=(10, 8))
    plt.contourf(x_centers, y_centers, hist.T, levels=400, cmap='inferno')
    plt.colorbar(label='Projected Density')
    #plt.title(f'2D Contour of Galactic Slice (Z in [{z_min}, {z_max}])', fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(hist_smoothed, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
    #            origin='lower', cmap='inferno')
    # plt.colorbar(label='Projected Density')
    # plt.title(f'2D Glowing Contour of Galactic Slice (Z in [{z_min}, {z_max}])', fontsize=14)
    # plt.xlabel('X Position', fontsize=12)
    # plt.ylabel('Y Position', fontsize=12)
    # plt.show()

def Galaxy_Mass_Dist():
    plt.figure(figsize=(10, 6))
    plt.hist(data["        M_star_tot [M_sol]"], bins=100, alpha=0.7, label="Stellar Mass (M_star)", log=True)
    plt.hist(data["M_DM_tot [M_sol]"], bins=100, alpha=0.7, label="Dark Matter Mass (M_DM)", log=True)
    plt.hist(data["M_Gas_tot [M_sol]"], bins=100, alpha=0.7, label="Gas Mass (M_gas)", log=True)

    # Add labels, legend, and title
    plt.xlabel("Mass (M_sol)", fontsize=12)
    plt.ylabel("Frequency (log scale)", fontsize=12)
    plt.title("Galaxy Mass Distribution", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()  

def sample_galaxies(n_samples, size, mass_threshold=1e6, visualize=False):
# Clean column names
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.strip()

    # Merge datasets on the correct keys
    merged_data = pd.merge(data, data, how='inner', left_on='#GalaxyID', right_on='1')

    # Add weights based on M_star_tot [M_sol]
    merged_data['weight'] = merged_data['M_star_tot [M_sol]'].apply(lambda x: x if x > mass_threshold else 1)
    merged_data['weight'] = merged_data['weight'] / merged_data['weight'].sum()  # Normalize weights

    # Perform weighted random sampling
    sampled_galaxies = merged_data.sample(n=n_samples, weights='weight', random_state=None)

    # Visualize if requested
    if visualize:
        x_sampled = sampled_galaxies['11.38892']
        y_sampled = sampled_galaxies['80.65985']
        z_sampled = sampled_galaxies['54.157307']

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_sampled, y_sampled, z_sampled, c='b', marker='^', alpha=0.6, s=size)
        ax.set_title("Sampled Galaxies in 3D Space", fontsize=14)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_zlabel("Z Position", fontsize=12)
        plt.show()
# Print all sampled galaxies
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(sampled_galaxies[['#GalaxyID', 'M_star_tot [M_sol]', 'weight']])

def galaxy_perturbation_contour(minimum_z, maximum_z, bins=100):
    """
    Generate a 2D slice and contour plot of galaxies and their offset positions within a Z-range.
    
    Parameters:
        minimum_z (float): Minimum Z-coordinate for the slice.
        maximum_z (float): Maximum Z-coordinate for the slice.
        bins (int): Number of bins for the 2D histogram.
    """
    global galaxy_positions, perturbed_positions
    
    # Check if galaxies and perturbed positions are available
    if galaxy_positions is None or perturbed_positions is None:
        print("No galaxies and perturbed positions found. Please run option 7 first.")
        return
    
    # Extract Z-coordinates of galaxies and their offsets
    galaxy_z = np.array([g[2] for g in galaxy_positions])
    perturbed_z = np.array([b[2] for b in perturbed_positions])
    
    # Create a mask for galaxies and offsets within the Z-range
    galaxy_mask = (galaxy_z >= minimum_z) & (galaxy_z <= maximum_z)
    perturbed_mask = (perturbed_z >= minimum_z) & (perturbed_z <= maximum_z)
    
    # Extract X and Y coordinates for galaxies and offsets within the Z-range
    galaxy_x = np.array([g[0] for g in galaxy_positions])[galaxy_mask]
    galaxy_y = np.array([g[1] for g in galaxy_positions])[galaxy_mask]
    perturbed_x = np.array([b[0] for b in perturbed_positions])[perturbed_mask]
    perturbed_y = np.array([b[1] for b in perturbed_positions])[perturbed_mask]
    
    # Step 2: Create a 2D histogram of galaxy positions
    hist_galaxy, xedges, yedges = np.histogram2d(galaxy_x, galaxy_y, bins=bins, range=[[0, 100], [0, 100]])
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    
    # Step 3: Create a 2D histogram of perturbed positions
    hist_perturbed, _, _ = np.histogram2d(perturbed_x, perturbed_y, bins=bins, range=[[0, 100], [0, 100]])
    
    # Step 4: Plot the 2D slice of galaxy positions
    plt.figure(figsize=(10, 8))
    plt.scatter(galaxy_x, galaxy_y, s=5, alpha=0.7, color='red', label='Galaxy Positions')
    plt.scatter(perturbed_x, perturbed_y, s=5, alpha=0.7, color='blue', label='Offset Positions')
    plt.title(f'2D Slice of Galaxy and Offset Positions (Z in [{minimum_z}, {maximum_z}])', fontsize=14)
    plt.xlabel('X Position (Mpc)', fontsize=12)
    plt.ylabel('Y Position (Mpc)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Step 5: Plot the 2D contour of galaxy positions
    plt.figure(figsize=(10, 8))
    plt.contourf(x_centers, y_centers, hist_galaxy.T, levels=50, cmap='Reds', alpha=0.7, label='Galaxy Density')
    plt.colorbar(label='Galaxy Density')
    plt.contourf(x_centers, y_centers, hist_perturbed.T, levels=50, cmap='Blues', alpha=0.5, label='Offset Density')
    plt.colorbar(label='Offset Density')
    plt.title(f'2D Contour of Galaxy and Offset Positions (Z in [{minimum_z}, {maximum_z}])', fontsize=14)
    plt.xlabel('X Position (Mpc)', fontsize=12)
    plt.ylabel('Y Position (Mpc)', fontsize=12)
    plt.legend()
    plt.show()

def compute_distances_and_velocities(galaxy_positions, redshifts):

    distances = np.array([np.sqrt(g[0]**2 + g[1]**2 + g[2]**2) for g in galaxy_positions])
    velocities = np.array([z * 3e5 for z in redshifts])  # Convert redshift to km/s
    return distances, velocities



loading_screen()
play_sound(BOOT_SOUND)

while True:
    
   animated_menu()

   try:
       option = int(input('\nSelect an option: '))
       
       if option not in range(1,12):
           raise ValueError
       play_sound(OPTION_SOUND)
       if option == 1:
           Galaxy_Cluster()
          
       elif option == 2:
           z_min = float(input('Enter Z coordinate for slice start:'))
           z_max = float(input('Enter Z coordinate for slice end:'))
           Galaxy_Contour(z_min, z_max)
           print("____________________________\n\n")   
        
           

       elif option == 3:
           Galaxy_Mass_Dist()

       elif option == 4:
           n = int(input('Enter the sample size:'))
           s = float(input('Enter the size of the marker:'))
           if n > len(data):
            print(f"Sample size too large. Maximum allowable size is {len(data)}.")
           else:
            sample_galaxies(n, s, visualize=True)
           
       elif option == 5:
           main()
       elif option == 6:
           # Step 1: Select a random galaxy
           np.random.seed(42)
           
           
        
           galaxy_pos = g_vector_galaxy()
           b = generate_b(galaxy_pos)
           print(f"Selected Galaxy Position (g): {galaxy_pos}")
           print(f"Perturbed Position (b): {b}")

           plot_g_and_b_vectors(galaxy_pos, b)
# Step 2: Perturb the galaxy position

    #    elif option == 7:
    #        n_galaxies = int(input("Enter the number of galaxies:"))  # Number of galaxies to select
    #        global galaxy_positions, perturbed_positions
    #        galaxy_positions = select_multiple_galaxies(n_galaxies)
    
    # # Generate perturbed positions for each galaxy
    #        perturbed_positions = generate_multiple_b_vectors(galaxy_positions)
    #        print("Galaxy Positions and Offsets:")
    #        print("Galaxy ID | Galaxy Position (x, y, z) | Offset Position (x, y, z)")
    #        for i, (g, b) in enumerate(zip(galaxy_positions, perturbed_positions)):
    #           print(f"Galaxy {i+1}: {g} | {b}")
    #        data = {
    #            "Galaxy ID": [f"Galaxy {i+1}" for i in range(n_galaxies)],
    #             "Galaxy Position X": [g[0] for g in galaxy_positions],
    #             "Galaxy Position Y": [g[1] for g in galaxy_positions],
    #             "Galaxy Position Z": [g[2] for g in galaxy_positions],
    #             "Offset Position X": [b[0] for b in perturbed_positions],
    #             "Offset Position Y": [b[1] for b in perturbed_positions],
    #             "Offset Position Z": [b[2] for b in perturbed_positions],
    #         }
    #        os.makedirs(output_directory, exist_ok=True)
    #        df = pd.DataFrame(data)
    #        output_file = os.path.join(output_directory, f"galaxy_positions_and_offsets_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    #        df.to_csv(output_file, index=False)
    #        print(f"Data saved to {output_file}")
    # # Plot all galaxies and their perturbed positions
    #        plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions)

       elif option == 7:
           n_galaxies = int(input("Enter the number of galaxies:"))
           global galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list
           galaxy_positions = select_multiple_galaxies(n_galaxies)
           perturbed_positions = generate_multiple_b_vectors(galaxy_positions)
           d_true_list = []
           d_gw_samples_list = []
           for g in galaxy_positions:
               d_true, d_gw_samples = calculate_distances(g[0], g[1], g[2])
               d_true_list.append(d_true)
               d_gw_samples_list.append(d_gw_samples)
           plot_multiple_g_and_b_vectors(galaxy_positions, perturbed_positions, d_true_list, d_gw_samples_list)
            
        
       elif option == 8:
           if galaxy_positions is None or perturbed_positions is None:
               print("No galaxies selected. Please select galaxies first by running option 7.")
           else:
               minimum_z = float(input('Enter Z coordinate for slice start:'))
               maximum_z = float(input('Enter Z coordinate for slice end:'))
               galaxy_perturbation_contour(minimum_z, maximum_z)
               print("____________________________\n\n")   
        
       elif option == 9:
           estimate_hubble_constant()    
       elif option == 10:
           play_sound(EXIT_SOUND)
           ending_screen()
           break


   except ValueError:
       print("Invalid Option, Try Again \n")
       continue