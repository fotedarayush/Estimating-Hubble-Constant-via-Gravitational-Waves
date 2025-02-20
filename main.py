from galaxy_io import GalaxyIO
from galaxy_menu import GalaxyMenu
from galaxy_constants import FILE_PATH_1, FILE_PATH_2

# Load data and create Galaxy objects
positions_data, attributes_data = GalaxyIO.load_data(FILE_PATH_1, FILE_PATH_2)
if positions_data is None or attributes_data is None:
    print("Failed to load data. Exiting program.")
    exit(1)

galaxies = GalaxyIO.create_galaxies(positions_data, attributes_data)

# Run the menu
menu = GalaxyMenu(galaxies)
menu.run()