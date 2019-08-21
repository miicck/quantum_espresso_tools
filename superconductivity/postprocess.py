import os
from subprocess import check_output
from quantum_espresso_tools import parser
from scipy.optimize import curve_fit
import numpy as np

RY_TO_K = 157887.6633481157

# Model of the superconducting gap vs temperature
# used to fit for Tc
def gap_model(t, tc, gmax):
    t = [min(ti,tc) for ti in t]
    return gmax * np.tanh(1.74*np.sqrt(tc/t - 1))

# Get superconductivity info from eliashhberg function
def get_tc_info(omega, a2f, mu):

    # Use elk to solve the eliashberg equations
    # carry out caclulation in temporary directory
    elk_base_dir = check_output(["which", "elk"])
    elk_base_dir = elk_base_dir.decode("utf-8").replace("/src/elk\n", "")
    species_dir  = elk_base_dir+"/species/"

    # Create elk input file
    os.system("mkdir tmp_elk 2>/dev/null")
    elkin = open("tmp_elk/elk.in", "w")
    elkin.write("tasks\n260\n\nntemp\n20\nmustar\n{0}\n".format(mu))
    elkin.write("sppath\n'{0}'\n".format(species_dir))
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
    print("Solving eliashberg equations with mu = {0} ...".format(mu))
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

# List all files in the given folder
def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            yield os.path.join(root, filename)

# Find all a2F.dos* files in base_dir
# or its subdirectories and create a
# matching a2f.dos*.tc file containing tc
# info
def process_all_a2f(base_dir, overwrite=False):

    for f in listfiles(base_dir):

        if not "a2f.dos" in f.lower(): continue
        if f.endswith(".tc"): continue

        ftc = f+".tc"
        if os.path.isfile(ftc) and (not overwrite):
            print("Refusing to overwrite "+ftc)
            continue

        d = "/".join(f.split("/")[0:-1])

        print("Parsing a2F for "+f)
        omega, a2f, a2fnn, a2fp = parser.parse_a2f(f)
        if min(omega) < 0:
            print("Could not get tc for {0}, imaginary modes detected.".format(f))
            continue

        print("Getting T_c for "+f)
        w = open(ftc,"w")

        tc, lam, wav, tc_ad = get_tc_info(omega, a2f, 0.1)
        w.write("mu = 0.1\n")
        fs = "{0} # Tc (Eliashberg)\n{1} # Tc (Allen-Dynes)\n{2} # Lambda\n{3} # <w>\n"
        w.write(fs.format(tc,tc_ad,lam,wav))

        tc, lam, wav, tc_ad = get_tc_info(omega, a2f, 0.15)
        w.write("mu = 0.15\n")
        fs = "{0} # Tc (Eliashberg)\n{1} # Tc (Allen-Dynes)\n{2} # Lambda\n{3} # <w>\n"
        w.write(fs.format(tc,tc_ad,lam,wav))

        w.close()
