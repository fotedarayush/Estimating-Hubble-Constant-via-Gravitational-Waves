import pandas as pd
from galaxy import Galaxy
from galaxy_constants import FILE_PATH_1, FILE_PATH_2

class GalaxyIO:
    @staticmethod
    def load_data(FILE_PATH_1, FILE_PATH_2):
        """Load galaxy positions and attributes data."""
        positions_data = pd.read_csv(FILE_PATH_1)
        attributes_data = pd.read_csv(FILE_PATH_2)
        positions_data.columns = positions_data.columns.str.strip()
        attributes_data.columns = attributes_data.columns.str.strip()
        return positions_data, attributes_data

    @staticmethod
    def create_galaxies(positions_data, attributes_data):
        """Create a list of Galaxy objects from the data."""
        galaxies = []
        for i in range(len(positions_data)):
            x = positions_data.iloc[i, 2]
            y = positions_data.iloc[i, 3]
            z = positions_data.iloc[i, 4]
            stellar_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_star_tot [M_sol]")]
            dm_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_DM_tot [M_sol]")]
            gas_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_Gas_tot [M_sol]")]
            redshift = attributes_data.iloc[i, attributes_data.columns.get_loc("Redshift")]
            galaxy = Galaxy(x, y, z, stellar_mass, dm_mass, gas_mass, redshift)
            galaxies.append(galaxy)
        return galaxies

    # @staticmethod
    # def create_galaxies(positions_data, attributes_data):
    #     """Create a list of Galaxy objects from the data."""
    #     galaxies = []
    #     for i in range(len(positions_data)):
    #         x = positions_data.iloc[i, 2]
    #         y = positions_data.iloc[i, 3]
    #         z = positions_data.iloc[i, 4]
    #         stellar_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_star_tot [M_sol]")]
    #         dm_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_DM_tot [M_sol]")]
    #         gas_mass = attributes_data.iloc[i, attributes_data.columns.get_loc("M_Gas_tot [M_sol]")]
    #         redshift = attributes_data.iloc[i, attributes_data.columns.get_loc("Redshift")]
    #         galaxy = Galaxy(x, y, z, stellar_mass, dm_mass, gas_mass, redshift)
    #         galaxies.append(galaxy)
    #     return galaxies