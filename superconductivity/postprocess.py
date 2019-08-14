import os
import numpy as np

# Model of the superconducting gap vs temperature
# used to fit for Tc
def gap_model(t, tc, gmax):
        t = [min(ti,tc) for ti in t]
        return gmax * np.tanh(1.74*np.sqrt(tc/t - 1))

# Get superconductivity info from eliashhberg function
def get_tc_info(omega, a2f, mu):

        # Use elk to solve the eliashberg equations
        # carry out caclulation in temporary directory

        # Create elk input file
        os.system("mkdir tmp_elk 2>/dev/null")
        elkin = open("tmp_elk/elk.in", "w")
        elkin.write("tasks\n260\n\nntemp\n20\n")
        elkin.write("sppath\n'/rscratch/mjh261/elk-6.2.8/species/'\n")
        elkin.write("atoms\n1\n'La.in'\n1\n0 0 0 0 0 0\n")
        elkin.write("avec\n1 0 0\n0 1 0\n0 0 1")
        elkin.close()

        # Create a2F file
        a2fin = open("tmp_elk/ALPHA2F.OUT", "w")
        for w, a in zip(omega, a2f):
                w *= 0.5 # Convert Ry to Ha
                if a < 0: a = 0
                a2fin.write("{0} {1}\n".format(w,a))
        a2fin.close()

        # Run elk
        print("Solving eliashberg equations ...")
        os.system("cd tmp_elk; elk > /dev/null")

        # Read superconducting gap vs temperature from output
        gapf = open("tmp_elk/ELIASHBERG_GAP_T.OUT")
        lines = gapf.read().split("\n")
        gapf.close()

        ts   = []
        gaps = []
        for l in lines:
                vals = [float(w) for w in l.split()]
                if len(vals) != 3: continue
                ts.append(vals[0])
                gaps.append(vals[1])

        # Use Allen-Dynes equation to estimate Tc
        wa    = [[w, a] for w, a in zip(omega, a2f) if w > 0]
        lam   = np.trapz([2*a/w for w, a in wa], x=[w for w,a in wa])
        wav   = np.exp((2/lam)*np.trapz([np.log(w)*a/w for w, a in wa], x=[w for w,a in wa]))
        wav  *= RY_TO_K
        tc_ad = (wav/1.20)*np.exp(-1.04*(1+lam)/(lam-mu-0.62*lam*mu))

        # Fit to model to extract Tc from gap equations
        p0 = [tc_ad, max(gaps)] # Initial param guess from A-D
        par, cov = curve_fit(gap_model, ts, gaps, p0)
        print("Tc = {0} +/- {1} (Eliashberg) {2} (Allen-Dynes)".format(par[0], cov[0][0]**0.5, tc_ad))
        if np.isfinite(cov[0][0]): tc = par[0]
        else: tc = 0

        # Remove temporary directory
        os.system("rm -r tmp_elk")

        return [tc, lam, wav, tc_ad]
