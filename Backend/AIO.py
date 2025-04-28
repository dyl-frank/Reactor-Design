import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os 
from scipy.constants import N_A, pi
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.gridspec import GridSpec

import numpy as np

'''
    UNITS:
        mass_density: g/cm3
        atom_density: N/b-cm
        area: cm2
        cross sections: b
        power: W
        all dimensions: m
        molar_mass: g/mol

    Values for composition are (ZAID, volume fraction)
'''
def create_data():
    global data
    data = {
        "unit_cell" : {
            "area" : 0,
            "xs" : {}
        },

        "fission_spectrum" : np.array([0.365, 0.396, 0.173, 0.050, 0.012, 0.003, 0.001, 0.0]),

        "cells": {
            "fuel":{
                "mass_density": 19.0,
                "atom_density" : None,
                "area": pi * ((6.35/2)-0.04)**2,
                "composition": [
                ],
                "flux_disadvantage_factor" : np.array([1.60, 1.60, 1.60, 1.45, 1.45, 1.34, 1.34, 1.34])
            },

            "coolant":{
                "mass_density": 11.35,
                "atom_density": 3.299e-2, # PNNL
                "area": None,
                "composition": [
                    ("82000", 1)
                ],
                "flux_disadvantage_factor" : np.array([0.90, 0.90, 0.91, 0.91, 0.91, 0.94, 0.94, 0.94]),
            },

            "cladding":{
                "mass_density": 8,
                "atom_density" : 8.655e-2, # PNNL
                "area": np.pi *( (6.35/2)**2 - ((6.35/2)-0.04)**2),
                "composition": [
                    ("SS316", 1)
                ],
                "flux_disadvantage_factor" : np.array([1.01, 1.01, 1.01, 1.02, 1.02, 1.03, 1.03, 1.03])
            },


        },
        
        "energy_structure": {
            "lethargy_width": np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, np.log(750/1e-6)]),  # in eV
            "lower_energy_bounds": np.array([2.2e6, 820e3, 300e3, 110e3, 40e3, 15e3, 750, 0])
        },

        "nuclides": {
            "82000": {
                "molar_mass": 206.14,
                "xs": {
                    "sigma_tr": np.array([1.5, 2.2, 3.6, 3.5, 4.0, 3.9, 7.3, 3.2]),
                    "sigma_ng": np.array([0.0050, 0.0002, 0.0004, 0.0010, 0.0010, 0.0010, 0.0090, 0.0080]),
                    "sigma_f": np.array([0.0 for _ in range(8)]),
                    "sigma_rs": np.array([0.623, 0.6908, 0.4458, 0.2900, 0.3500, 0.3000, 0.0400, 0.0000]),
                    "nu_sigma_f": None  # to be computed below
                },
                "yield": {
                    "nu_f": np.array([0 for _ in range(8)])
                },
                "scattering": {
                    "sigma_ss": np.array([
                        [0.0000, 0.5200, 0.0900, 0.0030, 0.0090, 0.0010, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.6900, 0.0000, 0.0004, 0.0004, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.4400, 0.0050, 0.0008, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.2900, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3500, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0400],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
                    ])
                }
            },

            "SS316": {
                "molar_mass": None,
                "xs": {
                    "sigma_tr": np.array([2.2, 2.1, 2.4, 3.1, 4.5, 6.1, 6.9, 10.4]),
                    "sigma_ng": np.array([0.0200, 0.0030, 0.0050, 0.0060, 0.0080, 0.0120, 0.0320, 0.0200]),
                    "sigma_f": np.array([0.0 for _ in range(8)]),
                    "sigma_rs": np.array([1.0108, 0.4600, 0.1200, 0.1400, 0.2800, 0.0700, 0.0400, 0.0]),
                    "nu_sigma_f": None  # to be computed below
                },
                "yield": {
                    "nu_f": np.array([0 for _ in range(8)])
                },
                "scattering": {
                    "sigma_ss": np.array([
                        [0.0000, 0.7500, 0.2000, 0.5000, 0.0100, 0.0008, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.3300, 0.1000, 0.0200, 0.0100, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.1200, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.1400, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2800, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0700, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0400],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
                    ])
                }
            },

            "92238": {
                "molar_mass": 238.05078826,

                "xs": {
                    "sigma_tr": np.array([4.3, 4.8, 6.3, 9.3, 11.0, 12.0, 13.1, 11.0]),
                    "sigma_ng": np.array([0.0100, 0.0900, 0.1100, 0.1500, 0.2600, 0.4700, 0.8400, 1.4700]),
                    "sigma_f": np.array([0.58, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
                    "sigma_rs": np.array([2.293, 1.490, 0.375, 0.293, 0.200, 0.090, 0.0100, 0.0000]),
                    "nu_sigma_f": None  # to be computed below
                },
                "yield": {
                    "nu_f": np.array([2.91, 2.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                },
                "scattering": {
                    "sigma_ss": np.array([
                        [0.0000, 1.2800, 0.7800, 0.2000, 0.0300, 0.0030, 0.0000, 0.0000],
                        [0.0000, 0.0000, 1.0500, 0.4200, 0.0100, 0.0100, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.3300, 0.0400, 0.0050, 0.0009, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.2900, 0.0030, 0.0005, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1800, 0.0200, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0900, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0100],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
                    ])
                },
            },

            "94239": {
                "molar_mass": 239.0521634,
                "xs": {
                    "sigma_tr": np.array([4.5, 5.1, 6.3, 8.6, 11.0, 13.0, 16.0, 31.8]),
                    "sigma_ng": np.array([0.0100, 0.0300, 0.1100, 0.2000, 0.3500, 0.5900, 1.9800, 8.5400]),
                    "sigma_f": np.array([1.85, 1.82, 1.60, 1.51, 1.60, 1.67, 2.78, 10.63]),
                    "sigma_rs": np.array([1.4950, 0.8260, 0.3709, 0.1905, 0.1500, 0.0900, 0.0100, 0.0000]),
                    "nu_sigma_f": None  # to be computed below
                },
                "yield": {
                    "nu_f": np.array([3.40, 3.07, 2.95, 2.90, 2.88, 2.88, 2.87, 2.87])
                },
                "scattering": {
                    "sigma_ss": np.array([
                        [0.0000, 0.6600, 0.6000, 0.1900, 0.0400, 0.0050, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.6400, 0.1500, 0.0300, 0.0060, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.3100, 0.0500, 0.0100, 0.0009, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.1800, 0.0100, 0.0005, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1300, 0.0200, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0900, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0100],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ])
                }
            }
        }
    }

    # Compute nu_sigma_f for each nuclide
    for nuc in data["nuclides"].values():
        xs = nuc["xs"]
        nu = nuc["yield"]["nu_f"]
        xs["nu_sigma_f"] = nu * xs["sigma_f"]

    # Establish upper energy bounds and bin widht
    data["energy_structure"]["upper_energy_bounds"] = data["energy_structure"]["lower_energy_bounds"] * np.exp(data["energy_structure"]["lethargy_width"])
    data["energy_structure"]["upper_energy_bounds"][-1]= data["energy_structure"]["lower_energy_bounds"][-2]
    data["energy_structure"]["energy_bin_width"] =  data["energy_structure"]["upper_energy_bounds"]-data["energy_structure"]["lower_energy_bounds"]

    return

def add_nuclides(cell, nuclide_list):
    # Normalize atom fractions
    total= sum([n for _,n in nuclide_list])
    data["cells"][cell]["composition"] = nuclide_list


    data["cells"][cell]["atom_density"] = 0

    
    for nuclide in nuclide_list:
        ZAID, frac = nuclide
        # compute atom density in N/b-cm
        data["cells"][cell]["atom_density"] += 1e-24*N_A* frac/total* data["cells"][cell]["mass_density"]/data["nuclides"][ZAID]["molar_mass"]

    return

def add_flux_disadvantage(cell):
    R = data["cells"][cell]["flux_disadvantage_factor"]
    
    for nuclide in data["cells"][cell]["composition"]:
        ZAID, _ = nuclide

        data["nuclides"][ZAID]["scattering"]["sigma_ss"] *= R
        for xs in data["nuclides"][ZAID]["xs"].values():
            xs *= R
            
    return


def convert_to_macroscopic():

    for cell in data["cells"].values():
        atom_density = cell["atom_density"]
    
        for nuclide in cell["composition"]:
            ZAID, frac = nuclide

            data["nuclides"][ZAID]["scattering"]["sigma_ss"] *= atom_density*frac
            for xs in data["nuclides"][ZAID]["xs"].values():
                xs *= atom_density *frac
    
    return

def construct_unit_cell(pitch, coolant_cell, fuel_cell_list =[], assembly_type = "hex", flux_correction = True):
    if assembly_type == "hex":
        a = pitch/np.sqrt(3)
        data["unit_cell"]["side_length"] = a
        cell_area = 3*np.sqrt(3)/2 * a**2
        
    elif assembly_type == "square":
        cell_area = pitch**2
        data["unit_cell"]["side_length"] = pitch

    elif assembly_type == "circle":
        r = pitch/2 
        cell_area = pi*r**2
        data["unit_cell"]["side_length"] = 0

    data["unit_cell"]["area"] = cell_area
    data["unit_cell"]["xs"] = {}  # Ensure fresh xs dict
    FVF = 0  # Fuel Volume Fraction

    for cell in fuel_cell_list:
        if flux_correction:
            add_flux_disadvantage(cell)

        volume_frac = data["cells"][cell]["area"] / cell_area
        FVF += volume_frac

        for nuclide in data["cells"][cell]["composition"]:
            ZAID, _ = nuclide

            # Initialize or accumulate scattering matrix
            sigma_ss = data["nuclides"][ZAID]["scattering"]["sigma_ss"] * volume_frac
            if "sigma_ss" in data["unit_cell"]["xs"]:
                data["unit_cell"]["xs"]["sigma_ss"] += sigma_ss
                
            else:
                data["unit_cell"]["xs"]["sigma_ss"] = sigma_ss.copy()

            # Initialize or accumulate other cross sections
            for label, xs in data["nuclides"][ZAID]["xs"].items():
                if label in data["unit_cell"]["xs"]:
                    data["unit_cell"]["xs"][label] += xs * volume_frac
                else:
                    data["unit_cell"]["xs"][label] = (xs * volume_frac).copy()

    for nuclide in data["cells"][coolant_cell]["composition"]:
        ZAID, _ = nuclide
        coolant_frac = 1 - FVF

        data["unit_cell"]["xs"]["sigma_ss"] += data["nuclides"][ZAID]["scattering"]["sigma_ss"] * coolant_frac

        for label, xs in data["nuclides"][ZAID]["xs"].items():
            data["unit_cell"]["xs"][label] += xs * coolant_frac

    data["unit_cell"]["diffusion_coefficient"] = 1 / (3 * data["unit_cell"]["xs"]["sigma_tr"])

def compute_geometric_buckling(height, radius, extrapolation_distance = 20.0):
    return (np.pi/(height+2*extrapolation_distance))**2 + (2.405/(radius+extrapolation_distance))**2


def N_max_hex(pitch, D, ax=None):
    """
    pitch = flat-to-flat distance of each hexagon = 2 * apothem
    D     = diameter of enclosing circle
    returns the number of full hexes (N), and if verbose, plots them.
    """
    # 1) compute side length from pitch
    side    = pitch / np.sqrt(3)  # since apothem = side*√3/2

    R = D / 2.0
    centers = []
    N = 0

    # 2) tighter axial-coordinate bounds
    q_max = int(np.floor((R/side)))
    r_max = int(np.floor((R/side)))

    # 3) generate centers and count
    for q in range(-q_max, q_max + 1):
        for r in range(-r_max, r_max + 1):
            x = 1.5 * side * q
            y = np.sqrt(3) * side * (r + q/2.0)
            if np.hypot(x, y) <= R - side + 1e-12:
                centers.append((x, y))
                N += 1

    # 4) optional plot
    if ax:
        # circle boundary
        circle = Circle((0, 0), R, facecolor="cyan", edgecolor='black',)
        ax.add_patch(circle)

        for cx, cy in centers:
            verts = [
                (cx + side * np.cos(np.pi/3 * k),
                 cy + side * np.sin(np.pi/3 * k))
                for k in range(6)
            ]
            hexagon = Polygon(verts, closed=True, facecolor='cyan', edgecolor='black')
            fuel     = Circle((cx, cy), 6.25/2, facecolor='red', edgecolor='none')
            ax.add_patch(hexagon)
            ax.add_patch(fuel)

        ax.set_aspect('equal')
        ax.set_xlim(-R - side, R + side)
        ax.set_ylim(-R - side, R + side)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"Core Construction")
        ax.grid(True, linestyle='--', linewidth=0.5)
        
    return N

def N_max_square(side, D, ax=None):
    """
    side = side length of each square
    D    = diameter of enclosing circle
    Returns the number of full squares that fit entirely inside the circle.
    If ax is provided, also plots the circle and squares.
    """
    R = D / 2.0
    half_diag = side * np.sqrt(2) / 2.0  # half the square's diagonal
    max_index = int(np.floor((R - half_diag) / side))
    
    centers = []
    N = 0
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            cx = i * side
            cy = j * side
            # ensure the square's corners lie inside the circle
            if np.hypot(cx, cy) <= R - half_diag + 1e-12:
                centers.append((cx, cy))
                N += 1

    if ax is not None:
        # draw circle boundary
        circle = Circle((0, 0), R, facecolor='cyan', edgecolor='black')
        ax.add_patch(circle)
        # draw each square
        for cx, cy in centers:
            rect = Rectangle(
                (cx - side/2, cy - side/2),
                side, side,
                facecolor='cyan',
                edgecolor='black'
            )
            ax.add_patch(rect)
            fuel = Circle((cx, cy), 6.25/2, facecolor='red', edgecolor='none')
            ax.add_patch(fuel)

        ax.set_aspect('equal')
        ax.set_xlim(-R - side, R + side)
        ax.set_ylim(-R - side, R + side)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Core Construction')
        ax.grid(True, linestyle='--', linewidth=0.5)

    return N

def N_max_circle(pitch, D_large, ax=None):
    r = pitch/2
    R = D_large / 2.0
    dx = 2 * r
    dy = r * np.sqrt(3)
    
    centers = []
    # number of rows above and below center
    max_row = int(np.floor((R - r) / dy))
    
    for i in range(-max_row, max_row + 1):
        y = i * dy
        offset = r if (i % 2 != 0) else 0.0
        # horizontal span available for center positions
        x_max = np.sqrt(max(0.0, (R - r)**2 - y**2))
        # compute j min/max so that x = offset + j*dx lies inside ±x_max
        j_min = int(np.floor((-x_max - offset) / dx))
        j_max = int(np.ceil((x_max - offset) / dx))
        for j in range(j_min, j_max + 1):
            x = offset + j * dx
            # ensure full circle fits: center distance <= R-r
            if x**2 + y**2 <= (R - r)**2 + 1e-12:
                centers.append((x, y))
    
    N = len(centers)
    
    if ax is not None:
        # plot large circle
        ax.add_patch(Circle((0, 0), R, facecolor='cyan', edgecolor='black'))
        # plot small circles
        for (cx, cy) in centers:
            ax.add_patch(Circle((cx, cy), r, facecolor='cyan', edgecolor='black'))
            fuel = Circle((cx, cy), 6.25/2, facecolor='red', edgecolor='none')
            ax.add_patch(fuel)

        ax.set_aspect('equal')
        ax.set_xlim(-R - r, R + r)
        ax.set_ylim(-R - r, R + r)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Core Construction')
        ax.grid(True, linestyle='--', linewidth=0.5)

    return N

def plot_flux_per_lethargy(flux, ax):
    # Bin edges
    lower = np.array(data["energy_structure"]["lower_energy_bounds"])
    upper = np.array(data["energy_structure"]["upper_energy_bounds"])

    # Build step-plot arrays
    #  - for “post” stepping: value holds until the *next* edge
    edges = np.flip(np.insert(lower, 0, [upper[0]]))
    flux_per_dU = np.flip(flux/data["energy_structure"]["lethargy_width"])

    y = np.concatenate([flux_per_dU, [flux_per_dU[-1]]])
    ax.step(edges, y, where='post')
    ax.set_xscale('symlog')
    ax.set_yscale('log')

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Flux per Unit Lethargy\n(n·cm⁻²·s⁻¹)", multialignment="center")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def plot_flux_energy_spectrum(flux, ax):
    # Bin edges
    lower = np.array(data["energy_structure"]["lower_energy_bounds"])
    upper = np.array(data["energy_structure"]["upper_energy_bounds"])

    # Build step-plot arrays
    #  - for “post” stepping: value holds until the *next* edge
    edges = np.flip(np.insert(lower, 0, [upper[0]]))
    flux_energy_spectrum = np.flip(flux/data["energy_structure"]["energy_bin_width"])

    y = np.concatenate([flux_energy_spectrum, [flux_energy_spectrum[-1]]])
    ax.step(edges, y, where='post')
    ax.set_xscale('symlog')
    ax.set_yscale('log')

    # ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron Flux\n(n·cm⁻²·s⁻¹·eV⁻¹)", multialignment="center" )
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def solve(pitch=8, enrichment=0.1, height=100, diameter =250, power = 1200e6, coolant_cell = "coolant", fuel_cells = ["fuel", "cladding"], assembly ="hex", flux_correction = False, generate_report = True):
    create_data()

    
    # Add fuel composition to fuel cell according to enrichment
    add_nuclides(cell="fuel", nuclide_list=[("92238", 1-enrichment), ("94239", enrichment)])

    # Convert cross sections to macroscopic
    convert_to_macroscopic() 

    # Construct unit cell
    construct_unit_cell(pitch=pitch,
                        coolant_cell=coolant_cell,
                        fuel_cell_list=fuel_cells,
                        assembly_type=assembly, 
                        flux_correction=flux_correction)
        

    # Compute geometric buckling
    Bg2 = compute_geometric_buckling(height=height, radius=diameter/2)
    
    # Prototype loss matrix with transpose of scatering matrix
    L = -np.copy(data["unit_cell"]["xs"]["sigma_ss"]).T

    # Compute total xs
    sigma_t = data["unit_cell"]["xs"]["sigma_f"]+data["unit_cell"]["xs"]["sigma_rs"]+data["unit_cell"]["xs"]["sigma_ng"]

    # Compute leakage term
    leakage = Bg2 * data["unit_cell"]["diffusion_coefficient"]
    
    # Combine leakage and total interaction termis into a a diagonal
    D = np.diag(leakage + sigma_t)

    # Compile final loss matrix
    L += D

    # solve for phi and keff
    phi_vector = np.linalg.solve(L, data["fission_spectrum"])
    
    k_eff = np.sum(phi*data["unit_cell"]["xs"]["nu_sigma_f"])

    # Determine scaling factor for phi
    volume = height/4 * pi * diameter**2

    E_fiss = 200*1.6022e-13
    production = np.dot(phi, data["unit_cell"]["xs"]["sigma_f"]) * E_fiss
    norm = (power)/production
    phi *= norm/volume

    flux = phi/data["energy_structure"]["energy_bin_width"]
    
    # To return more than just flux and keff
    if generate_report:
        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(12, 5))
        outer_gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)
        
        # Compute number of cells in core and plot on leftmost subplot
        ax_left = fig.add_subplot(outer_gs[0, 0])
        if assembly == "hex":
            N = N_max_hex(pitch=pitch, D=diameter, ax=ax_left)
        elif assembly == "square":
            N = N_max_square(side=pitch, D=diameter, ax=ax_left)
        elif assembly == "circle":
            N = N_max_circle(pitch=pitch, D_large=diameter, ax=ax_left)
        
        # Create two stacked plots in centered with shared x
        center_gs = outer_gs[0, 1].subgridspec(2, 1, hspace=0.1)
        ax_center_top = fig.add_subplot(center_gs[0, 0])
        ax_center_bot = fig.add_subplot(center_gs[1, 0], sharex=ax_center_top)

        plot_flux_per_lethargy(flux=phi, ax = ax_center_bot)
        plot_flux_energy_spectrum(flux= flux, ax = ax_center_top)
        ax_center_top.tick_params(labelbottom=False)
    
        pu_loading_mass = N*data["cells"]["fuel"]["mass_density"]*height*enrichment*data["cells"]["fuel"]["mass_density"] / 1e6
        fuel_loading_mass = N*data["cells"]["fuel"]["mass_density"]*height*data["cells"]["fuel"]["mass_density"]/ 1e6
        # print(pu_loading_mass)
        FVF = data["cells"]["fuel"]["area"]/data["unit_cell"]["area"]
        missing_vol_frac = 100*(1-N*data["unit_cell"]["area"]/(pi/4 * diameter**2))
        # Right: textbox
        ax_right = fig.add_subplot(outer_gs[0, 2])
        ax_right.axis('off')
        textbox = (
            "Design Parameters:\n"
            f"- Number of Fuel Elements: {N}\n"
            f"- Pu-239 Fraction: {100*enrichment}%\n"
            f"- Core Height: {height} cm\n"
            f"- Core Diameter: {diameter} cm\n"
            f"- Loaded Fuel Mass: {fuel_loading_mass:.2f} MT\n"
            f"- Loaded Pu-239 Mass: {pu_loading_mass:.2f} MT\n"
            f"- Fuel Volume Fraction in Cell: {FVF:.2f}%\n"
            f"- Flux Disadvantage Factors: {flux_correction}\n"
            f"- Power {power/1e6} MW\n"
            f"- Unaccounted Volume Fraction = {missing_vol_frac:.2f}$\n"
            f"- K-effective: {k_eff:.4f}\n"
        
            
        )
        ax_right.text(-0.2, 0.5, textbox, ha='left', va='center', wrap=True)
        plt.show()
    return k_eff, flux

# Solve…
k_eff, flux = solve(
    enrichment=0.0,
    height=100,
    diameter=220,
    pitch=6.25 * 1.25,
    assembly = "square",
    flux_correction=False,
    generate_report=True
)
k_eff, flux = solve(
    enrichment=0.0550,
    height=100,
    diameter=220,
    pitch=6.25 * 1.25,
    assembly = "hex",
    flux_correction=False,
    generate_report=True
)
k_eff, flux = solve(
    enrichment=0.0550,
    height=100,
    diameter= 220,
    pitch=6.25 * 1.25,
    assembly = "circle",
    flux_correction=False,
    generate_report=True
)

