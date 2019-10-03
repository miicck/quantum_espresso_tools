from quantum_espresso_tools.parser import parse_vc_relax, parse_phonon_dos
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

RY_TO_K   = 157887.6633481157
RY_TO_CMM = 109736.75775046606
RY_TO_MEV = 13.605698065893753*1000.0 

# Plot a density of states, (or partial density of states)
def plot_dos(ws, pdos, labels=None, fermi_energy=0):
    tot = np.zeros(len(pdos[0]))
    ws = np.array(ws) - fermi_energy
    for i, pd in enumerate(pdos):
        label = None
        if not labels is None:
                label = labels[i]
        plt.fill_betweenx(ws, tot, tot+pd, label=label)
        tot += pd
    if not labels is None:
        plt.legend()
    plt.axhline(0, color="black")

# Plot a bandstructure (optionally specifying a file
# with the indicies of the high symmetry points)
def plot_bands(qs, all_ws, ylabel, hsp_file=None, fermi_energy=0, resolve_band_cross=False):

    # Parse high symmetry points
    if hsp_file is None:
        xtick_vals  = [0, len(qs)]
        xtick_names = ["", ""] 
    else:
        lines = open(hsp_file).read().split("\n")
        xtick_vals = []
        xtick_names = []
        for l in lines:
            if len(l.split()) == 0: continue
            index, name = l.split()
            xtick_vals.append(int(index))
            xtick_names.append(name)

    # Find discontinuities in the path
    dc_pts = []
    for i in xtick_vals:
        for j in xtick_vals:
            if j >= i: continue
            if abs(i-j) == 1:
                dc_pts.append(i)

    # Attempt to sort out band crossings
    bands = np.array(all_ws)
    for iq in range(1, len(bands) - 1):
        if not resolve_band_cross: break

        # Extrapolate modes at iq+1 from modes at iq and iq-1
        extrap = []
        for im in range(0, len(bands[iq])):
            extrap.append(2*bands[iq][im]-bands[iq-1][im])

        # Swap iq+1'th bands around until they minimize
        # difference to extrapolated values
        swap_made = True
        while swap_made:

            swap_made = False
            for it in range(1, len(bands[iq])):

                # Dont swap bands which are of equal value
                if (bands[iq+1][it] == bands[iq+1][it-1]): continue

                # If the order of extrapolated bands at iq+1 is not equal
                # to the order of bands at iq+1, swap the bands after iq
                if (extrap[it] < extrap[it-1]) != (bands[iq+1][it] < bands[iq+1][it-1]):

                    for iqs in range(iq+1, len(bands)):
                        tmp = bands[iqs][it]
                        bands[iqs][it]   = bands[iqs][it-1]
                        bands[iqs][it-1] = tmp

                    swap_made = True

    bands = bands.T

    # Plot the bands between each successive pair
    # of discontinuities
    dc_pts.append(0)
    dc_pts.append(len(bands[0]))
    dc_pts.sort()
    for band in bands:
        for i in range(1, len(dc_pts)):
            s = dc_pts[i-1]
            f = dc_pts[i]
            plt.plot(range(s,f),band[s:f]-fermi_energy,color=np.random.rand(3))

    plt.axhline(0, color="black")
    plt.ylabel(ylabel)
    plt.xticks(xtick_vals, xtick_names)
    for x in xtick_vals:
        plt.axvline(x, color="black", linestyle=":")

    for x in dc_pts:
        plt.axvline(x, color="black")
        plt.axvline(x-1, color="black")

def plot_gibbs_vs_pressure(system_dirs):
    
    frel = None # Will hold a model for E(v)
    plt.rc("text", usetex=True) # Use LaTeX

    all_data = []
    for direc in system_dirs:
        if not os.path.isdir(direc): continue

        # Plot all the sub-directories with relax.out and phonon.dos files
        data = []
        for p_dir in os.listdir(direc):

            p_dir      = direc + "/" + p_dir
            relax_file = p_dir + "/relax.out"
            dos_file   = p_dir + "/phonon.dos"

            if not os.path.isfile(relax_file):
                relax_file = p_dir + "/primary_kpts/relax.out"
                if not os.path.isfile(relax_file):
                    relax_file = p_dir + "/aux_kpts/relax.out"
                    if not os.path.isfile(relax_file):
                        print("{0} does not exist, skipping...".format(relax_file))
                        continue

            if not os.path.isfile(dos_file):
                dos_file = p_dir + "/primary_kpts/phonon.dos"
                if not os.path.isfile(dos_file):
                    dos_file = p_dir + "/aux_kpts/phonon.dos"
                    if not os.path.isfile(dos_file):
                        print("{0} does not exist, skipping...".format(dos_file))
                        continue

            print(relax_file)
            print(dos_file)

            # Read in phonon density of states
            omegas, pdos = parse_phonon_dos(dos_file)
            dos          = np.sum(pdos, axis=0)

            # Read in thermodynamic quantities
            relax    = parse_vc_relax(relax_file)
            nat      = len(relax["atoms"])
            pressure = relax["pressure"]
            volume   = relax["volume"]/nat
            enthalpy = relax["enthalpy"]/nat

            # Normalize dos to # of phonon states
            dos = 3*nat*dos/np.trapz(dos, x=omegas)

            # Calculate zero-point energy
            wd = [[w,d] for w,d in zip(omegas, dos) if w > 0]
            zpe = np.trapz([0.5*w*d for w,d in wd], x=[w for w,d in wd])/nat

            # Caclulate occupational contribution to
            # phonon free energy at 300 K
            t   = 300.0 / RY_TO_K
            occ = t*np.trapz([d*np.log(1 - np.exp(-w/t)) for w,d in wd], x=[w for w,d in wd])/nat

            stable = True
            for w, d in zip(omegas, dos):
                if w >= -10e-10: continue
                if d <=  10e-10: continue
                stable = False
                break

            # Save data
            gibbs = zpe + enthalpy
            data.append([pressure, volume, gibbs, occ, stable])

        if len(data) == 0:
            print("No data for "+direc)
            continue

        # Get stable, unstable and all data
        data.sort()
        all_data.append([direc, data])

    for direc, data in all_data:
        try: pss, vss, ess, des = np.array([d[0:-1] for d in data if d[-1] > 0.1]).T
        except ValueError: pss, vss, ess, des = [np.array([]) for i in range(0,4)] 
        try: psu, vsu, esu, deu = np.array([d[0:-1] for d in data if d[-1] < 0.1]).T
        except ValueError: psu, vsu, esu, deu = [np.array([]) for i in range(0,4)]
        ps,  vs,  es,  de, stable = np.array(data).T

        if frel is None:

            # Fit this data to a cubic spline
            frel = CubicSpline(ps, es + de)

        label = direc
        if "c2m"    in label : label =  "$C_2m$"
        if "fm3m"   in label : label = r"$Fm\bar{3}m$" 
        if "r3m"    in label : label =  "$R3m$"
        if "p63mmc" in label : label =  "$P6_3/mmc$"
        if "cmcm"   in label : label =  "$cmcm$"

        # Plot gibbs free energy at 300K
        p = plt.plot(ps/10.0, (es - frel(ps) + de)*RY_TO_MEV, label=label)
        col = p[0].get_color()

        # Plot gibbs free energy at 0K
        plt.plot(ps/10.0, (es - frel(ps))*RY_TO_MEV, linestyle=":", color=col)
        plt.scatter(pss/10.0, (ess - frel(pss) + des)*RY_TO_MEV, color=col)
        plt.scatter(psu/10.0, (esu - frel(psu) + deu)*RY_TO_MEV, marker="x", color=col)

    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Gibbs free energy\n(meV/atom, relative)")
    plt.legend()
    plt.show()



