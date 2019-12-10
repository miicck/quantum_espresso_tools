from quantum_espresso_tools.parser import parse_vc_relax, parse_phonon_dos, parse_bands
from quantum_espresso_tools.fits import fit_birch_murnaghan
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

RY_TO_K   = 157887.6633481157
RY_TO_CMM = 109736.75775046606
RY_TO_MEV = 13.605698065893753*1000.0 
KBAR_AU3_TO_RY = 1/(5.0*29421.02648438959)

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
            index, name = l.split()[:2]
            xtick_vals.append(int(index))
            xtick_names.append(name)

    # Find discontinuities in the path
    dc_pts    = []
    to_remove = []
    for i in xtick_vals:
        for jindex, j in enumerate(xtick_vals):
            if j >= i: continue
            if abs(i-j) == 1:
                ni = xtick_names[xtick_vals.index(i)]
                nj = xtick_names[xtick_vals.index(j)]
                if ni != nj: dc_pts.append(i)
                else: to_remove.append([jindex,j])

    # Remove repeated q-points
    to_remove.sort(key=lambda i:-i[1])
    for i, iq in to_remove:
        del qs[iq]
        del all_ws[iq]

        # Move labels to compenstate for change
        for j in range(i, len(xtick_vals)):
            xtick_vals[j] = xtick_vals[j] - 1

    # Remove repeated labels
    to_remove.sort(key=lambda i:-i[0])
    for i, iq in to_remove:
        del xtick_vals[i]
        del xtick_names[i]

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

def plot_gibbs_vs_pressure_simple(system_dirs):
    plt.rc("text", usetex=True)
    
    all_data = []
    for direc in system_dirs:
        if not os.path.isdir(direc): continue

        # Plot all the sub-directories with relax.out and phonon.dos files
        data = []
        for p_dir in os.listdir(direc):

            p_dir      = direc + "/" + p_dir
            relax_file = p_dir + "/relax.out"
            dos_file   = p_dir + "/phonon.dos"

            # Read results of geometry optimization (try a few possible locations)
            if not os.path.isfile(relax_file):
                relax_file = p_dir + "/primary_kpts/relax.out"
                if not os.path.isfile(relax_file):
                    relax_file = p_dir + "/aux_kpts/relax.out"
                    if not os.path.isfile(relax_file):
                        print("{0} does not exist, skipping...".format(relax_file))
                        continue

            # Read phonon density of states (try a few possible locations)
            if not os.path.isfile(dos_file):
                dos_file = p_dir + "/primary_kpts/phonon.dos"
                if not os.path.isfile(dos_file):
                    dos_file = p_dir + "/aux_kpts/phonon.dos"
                    if not os.path.isfile(dos_file):
                        print("{0} does not exist, skipping...".format(dos_file))
                        continue

            # Read in phonon density of states
            omegas, pdos = parse_phonon_dos(dos_file)
            dos          = np.sum(pdos, axis=0)

            # Read in thermodynamic quantities
            relax    = parse_vc_relax(relax_file)
            nat      = len(relax["atoms"])
            pressure = relax["pressure"]
            volume   = relax["volume"]/nat
            enthalpy = relax["enthalpy"]/nat
            energy   = relax["energy"]/nat

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

            # Calculate Helmholtz free energy
            f0   = energy + zpe
            f300 = energy + zpe + occ

            # Save data
            data.append([volume, f0, f300, pressure, enthalpy, energy, stable])

        if len(data) == 0:
            print("No data for "+direc)
            continue

        # Sort data by inverse volume <=> pressure
        data.sort(key=lambda d:1/d[0])
        all_data.append([direc, data])

    grel300   = None
    hrel      = None
    greldft   = None
    ereldft   = None
    fig, axes = plt.subplots(2,2)

    for direc, data in all_data:
        v, f0, f300, p_dft, h_dft, e_dft, s = np.array(data).T

        if len(h_dft) == 1:
            pv = KBAR_AU3_TO_RY*p_dft[0]*v[0]
            print(direc, f300, f300+pv, pv)
            continue

        if hrel is None: hrel = CubicSpline(p_dft, h_dft)
        axes[0,0].plot(p_dft, (h_dft-hrel(p_dft))*RY_TO_MEV)
        axes[0,0].set_xlabel(r"$P_{DFT}$ (KBar)")
        axes[0,0].set_ylabel(r"Enthalpy (meV/atom)")

        use_unstable = True
        v_fit     = [i for i,j in zip(v,   s)  if j or use_unstable]
        f0_fit    = [i for i,j in zip(f0,  s)  if j or use_unstable]
        f300_fit  = [i for i,j in zip(f300,s)  if j or use_unstable]
        p_dft_fit = [i for i,j in zip(p_dft,s) if j or use_unstable] 

        label = direc
        if  "fm3m"  in label: label = r"$Fm\bar{3}m$"
        elif "c2m"  in label: label = r"$C_2/m$"
        elif "p63"  in label: label = r"$P6_3/mmc$"
        elif "cmcm" in label: label = r"$Cmcm$"

        e_model, p_model, par, cov = fit_birch_murnaghan(v_fit, f300_fit, p_guess=p_dft_fit)
        p300 = p_model(v)
        g300 = f300 + p300*v*KBAR_AU3_TO_RY
        if grel300 is None: grel300 = CubicSpline(p300, g300)
        p = axes[0,1].plot(p300, (g300-grel300(p300))*RY_TO_MEV, label=label)
        plt.legend()
        col = p[0].get_color()

        gs = [i for i,j in zip(g300, s) if j]
        ps = [i for i,j in zip(p300, s) if j]
        axes[0,1].scatter(ps, (gs-grel300(ps))*RY_TO_MEV, color=col)

        gu = [i for i,j in zip(g300, s) if not j]
        pu = [i for i,j in zip(p300, s) if not j]
        axes[0,1].scatter(pu, (gu-grel300(pu))*RY_TO_MEV, color=col, marker="x")

        e_model, p_model, par, cov = fit_birch_murnaghan(v_fit, f0_fit, p_guess=p_dft_fit)
        p0   = p_model(v)
        g0   = f0 + p0*v*KBAR_AU3_TO_RY
        axes[0,1].plot(p0, (g0-grel300(p0))*RY_TO_MEV, linestyle=":", color=col)
        axes[0,1].set_xlabel(r"$P$ (KBar)")
        axes[0,1].set_ylabel(r"Gibbs free energy (meV/atom)")

        g_dft300 = f300 + p_dft*v*KBAR_AU3_TO_RY
        if greldft is None: greldft = CubicSpline(p_dft, g_dft300)
        axes[1,0].plot(p_dft, (g_dft300-greldft(p_dft))*RY_TO_MEV)
        axes[1,0].set_xlabel(r"$P_{DFT}$ (KBar)")
        axes[1,0].set_ylabel(r"Gibbs free energy (DFT, meV/atom)")

        if ereldft is None: ereldft = CubicSpline(p_dft, e_dft)
        axes[1,1].plot(p_dft, (e_dft-ereldft(p_dft))*RY_TO_MEV)
        axes[1,1].set_xlabel(r"$P_{DFT}$ (KBar)")
        axes[1,1].set_ylabel(r"$E_{DFT}$ (meV/atom)")

    plt.show()

def plot_gibbs_vs_pressure(system_dirs):
    return plot_gibbs_vs_pressure_simple(system_dirs)
    
    all_data = []
    for direc in system_dirs:
        if not os.path.isdir(direc): continue

        # Plot all the sub-directories with relax.out and phonon.dos files
        data = []
        for p_dir in os.listdir(direc):

            p_dir      = direc + "/" + p_dir
            relax_file = p_dir + "/relax.out"
            dos_file   = p_dir + "/phonon.dos"

            # Read results of geometry optimization (try a few possible locations)
            if not os.path.isfile(relax_file):
                relax_file = p_dir + "/primary_kpts/relax.out"
                if not os.path.isfile(relax_file):
                    relax_file = p_dir + "/aux_kpts/relax.out"
                    if not os.path.isfile(relax_file):
                        print("{0} does not exist, skipping...".format(relax_file))
                        continue

            # Read phonon density of states (try a few possible locations)
            if not os.path.isfile(dos_file):
                dos_file = p_dir + "/primary_kpts/phonon.dos"
                if not os.path.isfile(dos_file):
                    dos_file = p_dir + "/aux_kpts/phonon.dos"
                    if not os.path.isfile(dos_file):
                        print("{0} does not exist, skipping...".format(dos_file))
                        continue

            # Read in phonon density of states
            omegas, pdos = parse_phonon_dos(dos_file)
            dos          = np.sum(pdos, axis=0)

            # Read in thermodynamic quantities
            relax    = parse_vc_relax(relax_file)
            nat      = len(relax["atoms"])
            pressure = relax["pressure"]
            volume   = relax["volume"]/nat
            enthalpy = relax["enthalpy"]/nat
            energy   = relax["energy"]/nat

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
            data.append([volume, energy, enthalpy, zpe, occ, pressure, stable])

        if len(data) == 0:
            print("No data for "+direc)
            continue

        # Sort data by inverse volume <=> pressure
        data.sort(key=lambda d:1/d[0])
        all_data.append([direc, data])

    hrel  = None
    erel  = None
    pvrel = None

    for direc, data in all_data:
        
        # Get the data for this system
        vs, es, enthalpy, zpes, occs, dft_ps, ss = np.array(data).T
        pvs = dft_ps*vs*KBAR_AU3_TO_RY

        if hrel is None:
            hrel = CubicSpline(dft_ps, enthalpy)

        plt.subplot(313)
        plt.plot(dft_ps, (enthalpy-hrel(dft_ps))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("Enthalpy (meV/atom)")

        if erel is None:
            erel = CubicSpline(dft_ps, es)

        plt.subplot(312)
        plt.plot(dft_ps, (es-erel(dft_ps))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("E (meV/atom)")

        if pvrel is None:
            pvrel = CubicSpline(dft_ps, pvs)

        plt.subplot(311)
        plt.plot(dft_ps, (pvs - pvrel(dft_ps))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("PV (meV/atom)")

    plt.figure()

    # Get the data at the phonon-corrected pressures
    corrected_data = []
    for direc, data in all_data:
        
        # Get the data for this system
        vs, es, enthalpy, zpes, occs, dft_ps, ss = np.array(data).T

        # Only fit to the stable points
        include_unstable = False
        vfit    = [v for v, s in zip(vs, ss)           if s or include_unstable]
        pfit    = [p for p, s in zip(dft_ps, ss)       if s or include_unstable]
        efit0   = [e for e, s in zip(es+zpes, ss)      if s or include_unstable]
        efit300 = [e for e, s in zip(es+zpes+occs, ss) if s or include_unstable]

        # Fit the birch murnaghan E.O.S at 0 K
        # and calculate the gibbs as a function of corrected pressure
        e_model, p_model, par, cov = fit_birch_murnaghan(vfit, efit0, p_guess=pfit)
        ps_corrected_0K = p_model(vs)
        #gibbs_0K        = es + zpes + KBAR_AU3_TO_RY*ps_corrected_0K*vs 
        #gibbs_0K_dft    = enthalpy + zpes

        # Fit the birch murnaghan E.O.S at 300 K
        # and calculate the gibbs as a function of corrected pressure
        e_model, p_model, par, cov = fit_birch_murnaghan(vfit, efit300, p_guess=pfit)
        ps_corrected_300K = p_model(vs)
        #gibbs_300K        = es + zpes + occs + KBAR_AU3_TO_RY*ps_corrected_300K*vs
        #gibbs_300K_dft    = enthalpy + zpes + occs

        # Re-evaluate quantities on the corrected-pressure grid
        es    = CubicSpline(dft_ps, es)
        es0   = es(ps_corrected_0K)
        es300 = es(ps_corrected_300K)

        zpes    = CubicSpline(dft_ps, zpes)
        zpes0   = zpes(ps_corrected_0K)
        zpes300 = zpes(ps_corrected_300K)

        occs    = CubicSpline(dft_ps, occs)
        occs0   = occs(ps_corrected_0K)
        occs300 = occs(ps_corrected_300K)

        # Store the results
        corrected_data.append([
            direc, 
            vs,
            es0,
            es300,
            zpes0,
            zpes300,
            occs0,
            occs300,
            dft_ps,
            ps_corrected_300K, 
            ps_corrected_0K, 
            ss
        ])

        plot_fitting_info = False
        if plot_fitting_info:

            plt.subplot(221)
            plt.plot(vs, (es + zpes - e_model(vs))*RY_TO_MEV)
            plt.ylabel("Birch murnaghan fit error (meV)")
            plt.xlabel("Volume/atom")

            plt.subplot(222)
            plt.plot(vs, ps_corrected_0K-dft_ps,   label="B-M pressure correction (0K)", linestyle=":")
            plt.plot(vs, ps_corrected_300K-dft_ps, label="B-M pressure correction (300K)")
            plt.ylabel("Pressure (KBar)")
            plt.xlabel("Volume/atom")
            plt.legend()

            plt.subplot(223)
            gp300 = CubicSpline(ps_corrected_300K, gibbs_300K)
            gp0   = CubicSpline(ps_corrected_0K,   gibbs_0K)
            plt.plot(dft_ps, (gp300(dft_ps) - gibbs_300K_dft)*RY_TO_MEV, label="300 K")
            plt.plot(dft_ps, (gp0(dft_ps)   - gibbs_0K_dft  )*RY_TO_MEV, label="0 K"  )
            plt.xlabel("Pressure (KBar)")
            plt.ylabel("Correction to \n Gibbs free energy/atom (meV)")
            plt.legend()

            plt.figure()

    frel  = None
    erel  = None
    zrel  = None
    orel  = None
    pvrel = None

    for direc, vs, es0, es300, zpe0, zpe300, occ0, occ300, dft_ps, p300, p0, ss in corrected_data:
        
        pvs300  = KBAR_AU3_TO_RY*p300*vs
        pvs0    = KBAR_AU3_TO_RY*p0*vs
        g300    = es300 + zpe300 + occ300 + pvs300
        g0      = es0 + zpe0 + pvs0

        label = direc
        if "c2m"    in label : label = r"$C_2/m$"
        if "fm3m"   in label : label = r"$Fm\bar{3}m$" 
        if "r3m"    in label : label =  "$R3m$"
        if "p63mmc" in label : label =  "$P6_3/mmc$"
        if "cmcm"   in label : label =  "$cmcm$"

        if erel is None: erel = CubicSpline(p300, es300)
        plt.subplot(511)
        plt.plot(p300, (es300-erel(p300))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("E (meV/atom)")

        if zrel is None: zrel = CubicSpline(p300, zpe300)
        plt.subplot(512)
        plt.plot(p300, (zpe300-zrel(p300))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("Z.P.E (meV/atom)")

        if orel is None: orel = CubicSpline(p300, occ300)
        plt.subplot(513)
        plt.plot(p300, (occ300-orel(p300))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("Phonon occupation energy (meV/atom)")

        if pvrel is None: pvrel = CubicSpline(p300, pvs300)
        plt.subplot(514)
        plt.plot(p300, (pvs300-pvrel(p300))*RY_TO_MEV)
        plt.xlabel("Pressure (KBar)")
        plt.ylabel("PV(meV/atom)")

        if frel is None: 
            # Plot relative to cubic spline fit of first system
            frel = CubicSpline(p300, g300)

        # Plot gibbs free energy at 300K
        plt.subplot(515)
        p = plt.plot(p300/10.0, (g300 - frel(p300))*RY_TO_MEV, label=label)
        col = p[0].get_color()

        # Plot gibbs free energy at 0K
        #plt.plot(p0/10.0, (g0 - frel(p0))*RY_TO_MEV, linestyle=":", color=col)

        # Label stable points
        p300s = np.array( [p for p, s in zip(p300, ss) if s] )
        g300s = np.array( [g for g, s in zip(g300, ss) if s] )
        plt.scatter(p300s/10.0, (g300s - frel(p300s))*RY_TO_MEV, color=col)

        p300u = np.array( [p for p, s in zip(p300, ss) if not s] )
        g300u = np.array( [g for g, s in zip(g300, ss) if not s] )
        plt.scatter(p300u/10.0, (g300u - frel(p300u))*RY_TO_MEV, color=col, marker="x")

    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Gibbs free energy\n(meV/atom, relative)")
    plt.legend()
    plt.show()



