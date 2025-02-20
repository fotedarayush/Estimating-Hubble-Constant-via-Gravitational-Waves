import numpy as np

class GalaxyPerturbation:
    @staticmethod
    def orthogonal_unit_vectors(g):
        """Generate two orthogonal unit vectors to the input vector g."""
        x = np.array([1, 0, 0])
        if np.allclose(g, x):
            x = np.array([0, 1, 0])
        u1 = np.cross(g, x)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(g, u1)
        u2 /= np.linalg.norm(u2)
        return u1, u2
    
    @staticmethod
    def generate_b(g, theta=None, phi=None, sigma=0.5):
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
        u1, u2 = GalaxyPerturbation.orthogonal_unit_vectors(g_norm)

        # Generate b vector
        r = u1 * np.sin(phi) + u2 * np.cos(phi)
        b = g_norm * np.cos(theta) + r * np.sin(theta)

        # Scale b by the magnitude of the original g vector
        b = b * np.linalg.norm(g)

        return b

@staticmethod
def generate_multiple_b_vectors(galaxy_positions, sigma=0.5, min_theta_deg=4, max_theta_deg=6):
    perturbed_positions = []
    for g in galaxy_positions:
        b = GalaxyPerturbation.generate_b(g, sigma=sigma)
        perturbed_positions.append(b)
    return perturbed_positions
        