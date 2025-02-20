import numpy as np

class Galaxy:
    def __init__(self, x, y, z, stellar_mass=None, dm_mass=None, gas_mass=None, redshift=None):
        self.x = x
        self.y = y
        self.z = z
        self.stellar_mass = stellar_mass
        self.dm_mass = dm_mass
        self.gas_mass = gas_mass
        self.redshift = redshift

    def position(self):
        """Return the position vector of the galaxy."""
        return np.array([self.x, self.y, self.z])

    def distance(self):
        """Calculate the distance of the galaxy from the origin."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def velocity(self):
        """Calculate the recessional velocity of the galaxy."""
        if self.redshift is None:
            raise ValueError("Redshift is not defined for this galaxy.")
        return self.redshift * 3e5  # Convert redshift to km/s