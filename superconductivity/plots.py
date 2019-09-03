import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from quantum_espresso_tools.parser import parse_vc_relax, parse_a2f

def convert_common_labels(label):
    if "c2m" in label: return "$C_2m$"
    if "fm3m" in label: return r"$Fm\bar{3}m$"
    if "p63mmc" in label: return "$P6_3/mmc$"
    return label

def plot_tc_vs_smearing(directories):
    
    for d in directories:
        plot_tc_vs_smearing_single(d, show=False)
    plt.show()

def plot_tc_vs_smearing_single(direc, show=True):

    if not os.path.isdir(direc): return
    mu_data = {}
    for f in os.listdir(direc):
        if not f.endswith(".tc"): continue
        f = direc+"/"+f
        lines = open(f).read().split("\n")

        isig = int(f.split(".")[-2].replace("dos",""))

        for i, l in enumerate(lines): 
            if "mu =" in l:
                mu = float(l.split("=")[-1])
                data = [isig]
                for j in range(i+1,i+5):
                    data.append(float(lines[j].split()[0]))
                if not mu in mu_data: mu_data[mu] = []
                mu_data[mu].append(data)

    sig_incr = 1.0
    elph_in = direc+"/elph.in"
    if os.path.isfile(elph_in):
        lines = open(elph_in).read().split("\n")
        for l in lines:
            if "el_ph_sigma" in l:
                sig_incr = float(l.split("=")[-1].replace(",",""))

    xlabel = "Smearing amount"
    if sig_incr != 1.0:
        xlabel = "Smearing width (Ry)"

    mus = []
    for mu in mu_data:
        mus.append(mu)
    if len(mus) == 0:
        return

    data1 = mu_data[mus[0]]
    data1.sort()
    data1 = np.array(data1).T

    data2 = mu_data[mus[1]]
    data2.sort()
    data2 = np.array(data2).T

    data1[0] *= sig_incr
    data2[0] *= sig_incr

    plt.subplot(211)
    p = plt.plot(data1[0], (data1[1] + data2[1])/2, marker="+", linestyle=":")
    plt.fill_between(data1[0], data1[1], data2[1], alpha=0.15, color=p[0].get_color(), label=direc)
    plt.xlabel(xlabel)
    plt.xlim([0, max(data1[0])])
    plt.ylabel("Tc (K) - Eliashberg \n $\mu^* \in [{0},{1}]$".format(*mus))
    plt.legend()

    plt.subplot(212)
    p = plt.plot(data1[0], (data1[2] + data2[2])/2, marker="+", linestyle=":")
    plt.fill_between(data1[0], data1[2], data2[2], alpha=0.15, color=p[0].get_color(), label=direc)
    plt.xlabel(xlabel)
    plt.xlim([0, max(data1[0])])
    plt.ylabel("Tc (K) - Mcmillan-Allen-Dynes \n $\mu^* \in [{0},{1}]$".format(*mus))
    plt.legend()

    if show: plt.show()

def plot_tc_vs_p(direc, show=True, plot_unstable=False):

    # Use LaTeX
    plt.rc("text", usetex=True)

    # Collect data for different pressures in this directory
    data = []
    for pdir in os.listdir(direc):

        # Get the pressure from the relax.out file
        relax_file = direc+"/"+pdir+"/relax.out"
        if not os.path.isfile(relax_file):
            print(relax_file+" does not exist, skipping...")
            continue

        relax = parse_vc_relax(relax_file)
        pressure = relax["pressure"]

        # Check if the a2F.tc file exists
        a2f_file = direc+"/"+pdir+"/a2F.dos10.tc"
        if not os.path.isfile(a2f_file):
            print(a2f_file+" does not exist, skipping...")
            continue

        # Read the a2F.tc file
        with open(a2f_file) as f:
            lines = f.read().split("\n")

        # Check if the structure is unstable
        unstable = lines[10].split("#")[0].strip() == "True"
        if unstable and (not plot_unstable): 
            continue

        # Read in mus and corresponding tcs
        mu1, mu2 = [float(l.split("=")[-1]) for l in [lines[0], lines[5]]]
        tc, tcad, tc2, tcad2 = [float(l.split("#")[0]) for l in [lines[1], lines[2], lines[6], lines[7]]]

        # Record the data
        data.append([pressure, tc, tc2, tcad, tcad2])

    if len(data) == 0:
        print("No data for "+direc+" skipping...")
        return

    # Transppose data into arrays for pressure, tc, tc2 ...
    data.sort()
    data = np.array(data).T
    data[0] /= 10 # Convert pressure from Kbar to GPa

    # Format label as math
    label = convert_common_labels(direc)

    # Plot Eliashberg Tc
    plt.subplot(211)
    plt.fill_between(data[0], data[1], data[2], alpha=0.5, label=label)
    plt.legend()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("$T_C$ (K)\nEliashberg, $\mu^* \in [{0},{1}]$".format(mu1, mu2))
    plt.subplot(212)

    # Plot Allen-Dynes Tc
    plt.fill_between(data[0], data[3], data[4], alpha=0.5, label=label)
    plt.legend()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("$T_C$ (K)\nAllen-Dynes, $\mu^* \in [{0},{1}]$".format(mu1, mu2))

    if show: plt.show()

def plot_a2f_vs_smearing(direcs):
    
    for d in direcs:
        plot_a2f_vs_smearing_single(d, show=False)
    plt.show()

def plot_a2f_vs_smearing_single(direc, show=True):

    f_elphin = direc+"/elph.in"
    if not os.path.isfile(f_elphin): return

    plt.rc("text", usetex=True)

    el_ph_sigma = 0.01
    with open(f_elphin) as f:
        lines = f.read().split()
        for l in lines:
            if "el_ph_sigma" in l:
                el_ph_sigma = float(l.split("=")[-1].replace(",",""))
                break

    data = []
    for f in os.listdir(direc):
        if not f.startswith("a2F.dos"): continue
        if f.endswith(".tc"): continue
        n = int(f.split(".")[-1].replace("dos",""))
        omega, a2f, a2fnn, a2fp = parse_a2f(direc+"/"+f)
        if min(omega) < 0:
            print("a2f.dos{0} is dynamically unstable".format(n))

        b = float(n)/10
        color = [b,1-b,0]
        sigma = el_ph_sigma * n
        aw = zip(a2f, omega)
        lam = 2.0*np.trapz([a/w for a, w in aw], x=omega)
        data.append([sigma, omega, a2f, color, lam])

    data.sort()
    plt.subplot(211)
    for sigma, omega, a2f, color, lam in data:
        plt.plot(omega, a2f, color=color, label="Smearing = {0} Ry".format(sigma))
    plt.legend()
    plt.xlabel("$\omega$ (Ry)")
    plt.ylabel("$a^2F(\omega)$")

    data = np.array(data).T
    plt.subplot(212)
    plt.plot(data[0], data[4], marker="+")
    plt.xlabel("$\sigma$ (Ry)")
    plt.ylabel(r"$\lambda = 2\int \frac{a^2F(\omega)}{w}d\omega$ (Ry)")

    if show: plt.show()

