import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from quantum_espresso_tools.parser import parse_vc_relax, parse_a2f

def convert_common_labels(label):
    if "c2m" in label: return "$C_2m$"
    if "fm3m" in label: return r"$Fm\bar{3}m$"
    if "p63mmc" in label: return "$P6_3/mmc$"
    if "cmcm" in label: return "$cmcm$"
    if "r3m" in label: return "$R3m$"
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

    label = direc
    if "kpq_" in label: label = "{0} k-points per q-point".format(int(label.split("kpq_")[-1])**3)
    if "_qpts" in label:
        n = label.split("_")[0]
        append = "(Primary)"
        if "aux" in label:
            append = "(Auxillary)"
        label = r"${0} \times {0} \times {0}$ ".format(n)+append

    plt.subplot(211)
    p = plt.plot(data1[0], (data1[1] + data2[1])/2, marker="+", linestyle=":")
    plt.fill_between(data1[0], data1[1], data2[1], alpha=0.15, color=p[0].get_color(), label=label)
    plt.xlabel(xlabel)
    plt.xlim([0, max(data1[0])])
    plt.ylabel("Tc (K) - Eliashberg \n $\mu^* \in [{0},{1}]$".format(*mus))
    plt.legend()

    plt.subplot(212)
    p = plt.plot(data1[0], (data1[2] + data2[2])/2, marker="+", linestyle=":")
    plt.fill_between(data1[0], data1[2], data2[2], alpha=0.15, color=p[0].get_color(), label=label)
    plt.xlabel(xlabel)
    plt.xlim([0, max(data1[0])])
    plt.ylabel("Tc (K) - Mcmillan-Allen-Dynes \n $\mu^* \in [{0},{1}]$".format(*mus))
    plt.legend()

    if show: plt.show()

def plot_tc_vs_p_aux_primary(
    sys_direc, 
    show=True, 
    plot_unstable=False, 
    plot_double_delta_info=False,
    plot_allen=False):
    
    # Plot Tc vs pressure for all of the pressure directories in
    # sys_direc, using primary and auxillary k-point grids to
    # estimate the correct double-delta smearing

    if not os.path.isdir(sys_direc):
        print("{0} is not a directory, skipping...".format(sys_direc))
        return

    sys_data = []

    # Loop over pressure directories
    for pdir in os.listdir(sys_direc):
        pdir = sys_direc + "/" + pdir
        if not os.path.isdir(pdir): continue

        grids_data = []
        i_grid_best = 0

        # Look over k-point grid directories
        for grid_dir in os.listdir(pdir):
            grid_dir = pdir + "/" + grid_dir
            if not os.path.isdir(grid_dir): continue

            if "primary" in grid_dir:
                i_grid_best = len(grids_data)

            relax   = None
            tc_data = []

            # Loop over files
            for filename in os.listdir(grid_dir):
                filename = grid_dir + "/" + filename

                if filename.endswith(".tc"):

                    # Parse tc information
                    isig = int(filename.split(".dos")[-1].split(".")[0])
                    with open(filename) as f:
                        lines = f.read().split("\n")
                        tc1 = float(lines[1].split("#")[0])
                        tc2 = float(lines[6].split("#")[0])

                    tc_data.append([isig, tc1, tc2])

                elif filename.endswith("relax.out"):
                    
                    # Parse vc-relax output
                    relax = parse_vc_relax(filename)

            if relax is None:
                print("Could not parse relaxation data in "+grid_dir)
                continue

            if len(tc_data) == 0:
                print("No Tc information found in "+grid_dir)
                continue

            tc_data.sort()
            sigma, tc1, tc2 = np.array(tc_data).T

            if plot_double_delta_info:
                label = "Kpoint grid {0}".format(len(grids_data)+1)

                plt.subplot(221)
                plt.plot(sigma, tc1, label=label)
                plt.xlabel("$\sigma (Ry)$")
                plt.ylabel("$T_C (K)$, Eliashberg\n$\mu^* = 0.1$")

                plt.subplot(222)
                plt.plot(sigma, tc2, label=label)
                plt.xlabel("$\sigma (Ry)$")
                plt.ylabel("$T_C (K)$, Eliashberg\n$\mu^* = 0.15$")

            grids_data.append({
                "relax"    : relax,
                "sigma"    : sigma,
                "tc1"      : tc1,
                "tc2"      : tc2,
                "pressure" : relax["pressure"]
            })

        if len(grids_data) < 2:
            print("Less than 2 k-point grids found in "+pdir+", skipping...")
            continue

        if len(grids_data[0]["tc1"]) != len(grids_data[1]["tc1"]):
            raise Exception("Mismatched grid sizes in "+pdir)

        # Evaluate the difference in Tc(sigma) between
        # the two grids and use this to work out what the
        # best smearing value is
        sigma = grids_data[0]["sigma"]
        dtc1  = list(abs(grids_data[0]["tc1"] - grids_data[1]["tc1"]))
        dtc2  = list(abs(grids_data[0]["tc2"] - grids_data[1]["tc2"]))
        dtc1 -= dtc1[-1]
        dtc2 -= dtc2[-1]

        delta_t_mu = np.mean(grids_data[0]["tc1"] - grids_data[0]["tc2"])

        # Find the best sigma <=> j by backtracking from
        # the largest smearing until the difference between
        # the two k-point grid reaches 10 K
        keep  = lambda dt : dt < 10

        for jbest1 in range(len(dtc1)-1, -1, -1):
            dt = dtc1[jbest1]
            if not keep(dt):
                jbest1 += 1
                break

        for jbest2 in range(len(dtc2)-1, -1, -1):
            dt = dtc2[jbest2]
            if not keep(dt):
                jbest2 += 1
                break

        tbest1 = grids_data[i_grid_best]["tc1"][jbest1]
        tbest2 = grids_data[i_grid_best]["tc2"][jbest2]

        tcmax = max(tbest1, tbest2)
        tcmin = min(tbest1, tbest2)
        tcav  = 0.5*(tcmax+tcmin)

        pressure  = np.mean([gd["pressure"] for gd in grids_data])
        dpressure = np.std([gd["pressure"] for gd in grids_data])
        sys_data.append([pressure, tcmin, tcmax])

        if plot_double_delta_info:
            plt.suptitle(r"$T_C = {0:8.2f} \pm {1:8.2f}$".format(tcav, (tcmax-tcmin)/2.0))

            plt.subplot(223)
            plt.plot(sigma, dtc1)
            plt.xlabel("$\sigma (Ry)$")
            plt.ylabel("$\Delta T_C (K)$, Eliashberg\n$\mu^* = 0.1$")
            plt.axvline(sigma[jbest1], color="green", label="Best $\sigma$")
            plt.legend()

            plt.subplot(221)
            plt.axvline(sigma[jbest1], color="green", label="Best $\sigma$")
            label = "Best $T_C \in [{0:8.2f}, {1:8.2f}]$"
            label = label.format(tcmin1, tcmax1)
            plt.axhspan(tcmin1, tcmax1, color="green", alpha=0.5, label=label)
            plt.legend()

            plt.subplot(224)
            plt.plot(sigma, dtc2)
            plt.xlabel("$\sigma (Ry)$")
            plt.ylabel("$\Delta T_C (K)$, Eliashberg\n$\mu^* = 0.15$")
            plt.axvline(sigma[jbest2], color="green", label="Best $\sigma$")
            plt.legend()

            plt.subplot(222)
            plt.axvline(sigma[jbest2], color="green", label="Best $\sigma$")
            label = "Best $T_C \in [{0:8.2f}, {1:8.2f}]$"
            label = label.format(tcmin2,  tcmax2)
            plt.axhspan(tcmin2, tcmax2, color="green", alpha=0.5, label=label)
            plt.legend()

            plt.tight_layout()
            plt.show()

    if len(sys_data) == 0:
        print("No data found for "+sys_direc)
        return

    sys_data.sort()
    pressure, tmin, tmax = np.array(sys_data).T
    label = convert_common_labels(sys_direc)
    plt.fill_between(pressure/10.0, tmin, tmax, alpha=0.5, label=label)
    plt.ylabel("$T_C$ (K, Eliashberg)\n"+r"$\mu^* \in [0.1, 0.15]$")
    plt.xlabel("Pressure (GPa)")
    plt.legend()
    if show: plt.show()

def get_best_a2f_dos_tc(direc):
    # Get the best a2F.dos{n}.tc file in the given
    # directory
    return "a2F.dos10.tc"

def plot_tc_vs_p(direc, 
    show=True, 
    plot_unstable=False, 
    plot_allen=False,
    plot_double_delta_info=False):

    # Use LaTeX
    plt.rc("text", usetex=True)

    if not os.path.isdir(direc):
        print("{0} is not a directory, skipping...".format(direc))
        return

    # Check to see if we're using a multi-grid scheme
    for pdir in os.listdir(direc):
        if not os.path.isdir(direc+"/"+pdir): continue
        for subdir in os.listdir(direc+"/"+pdir):
            if "aux_kpts" in subdir or "primary_kpts" in subdir:
                print("Using multi-grid scheme for "+direc)
                return plot_tc_vs_p_aux_primary(direc, show=show, 
                    plot_unstable=plot_unstable, plot_allen=plot_allen,
                    plot_double_delta_info=plot_double_delta_info)

    print("Using single-grid scheme for "+direc)

    # Collect data for different pressures in this directory
    data = []
    for pdir in os.listdir(direc):

        # Check this is a directory
        if not os.path.isdir(direc+"/"+pdir):
            continue

        # Get the pressure from the relax.out file
        relax_file = direc+"/"+pdir+"/relax.out"
        if not os.path.isfile(relax_file):
            print(relax_file+" does not exist, skipping...")
            continue

        relax = parse_vc_relax(relax_file)
        pressure = relax["pressure"]

        # Check if the a2F.tc file exists
        a2f_file = direc+"/"+pdir+"/"+get_best_a2f_dos_tc(direc+"/"+pdir)
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
    if plot_allen: plt.subplot(211)
    p = plt.plot(data[0], 0.5*(data[1]+data[2]), linestyle="none")
    color = p[0].get_color()
    plt.fill_between(data[0], data[1], data[2], alpha=0.5, label=label, color=color)
    plt.legend()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("$T_C$ (K)\nEliashberg, $\mu^* \in [{0},{1}]$".format(mu1, mu2))

    # Plot Allen-Dynes Tc
    if plot_allen:
        plt.subplot(212)
        plt.fill_between(data[0], data[3], data[4], alpha=0.5, label=label, color=color)
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

