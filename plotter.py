import matplotlib.pyplot as plt
import numpy as np
import sys
import os

RY_TO_K = 157887.6633481157
RY_TO_CMM = 109736.75775046606

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
