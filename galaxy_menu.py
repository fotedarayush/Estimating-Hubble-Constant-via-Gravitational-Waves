import time
import sys
from colorama import Fore
from playsound import playsound
from galaxy_constants import BOOT_SOUND, OPTION_SOUND, ERROR_SOUND, EXIT_SOUND

class GalaxyMenu:
    def __init__(self, galaxies):
        self.galaxies = galaxies

    def loading_screen(self):
        print('\nLoading GHOST', end='')
        for _ in range(11):
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(0.2)
        playsound(BOOT_SOUND)

    def animated_menu(self, first_run=True):
        options = [
            '1. 3D Galaxy Distribution Visualiser',
            '2. Sliced Distribution with Contour',
            '3. Galaxy Mass Distribution',
            '4. Random Galaxy Sampling',
            '5. Perform Error Box Simulation (1$\sigma$)',
            '6. Galaxy and its offset position',
            '7. Multiple galaxies with offset',
            '8. Galaxies and their offset contour slice',
            '9. Estimate the Hubble constant',
            '10. Exit'
        ]

        if first_run:
            print(Fore.GREEN + '\n\nWelcome to the Galaxy Hubble cOnstant eSTimator (GHOST)\n\n')
            time.sleep(0.2)
            for option in options:
                for char in option:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    time.sleep(0.02)
                print()
                time.sleep(0.15)
            print('\n\nSelect an option to continue')
        else:
            print(Fore.GREEN + '\n\nGalaxy Hubble cOnstant eSTimator (GHOST)\n\n')
            for option in options:
                print(option)
            print('\nSelect an option to continue')

    def run(self):
        self.loading_screen()
        while True:
            self.animated_menu(first_run=False)
            try:
                option = int(input('\nSelect an option: '))
                if option not in range(1, 11):
                    raise ValueError
                playsound(OPTION_SOUND)
                if option == 1:
                    GalaxyVisualizer.plot_3d_cluster(self.galaxies)
                elif option == 2:
                    z_min = float(input('Enter Z coordinate for slice start: '))
                    z_max = float(input('Enter Z coordinate for slice end: '))
                    GalaxyVisualizer.plot_contour(self.galaxies, z_min, z_max)
                elif option == 10:
                    playsound(EXIT_SOUND)
                    print('\nExiting the program. Goodbye!')
                    break
            except ValueError:
                print("Invalid Option, Try Again \n")
                playsound(ERROR_SOUND)