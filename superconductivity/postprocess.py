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
# we ignore the portion of a2f where w < 0 (if there is such a region)
def get_tc_info(omega, a2f, mu, plot_fit=False):

    # Use elk to solve the eliashberg equations
    # carry out caclulation in temporary directory
    elk_base_dir = check_output(["which", "elk"])
    elk_base_dir = elk_base_dir.decode("utf-8").replace("/src/elk\n", "")
    species_dir  = elk_base_dir+"/species/"

    # Create temporary directory to run elk in
    os.system("mkdir tmp_elk 2>/dev/null")

    # Create a2F file
    a2fin = open("tmp_elk/ALPHA2F.OUT", "w")
    wa    = [[w, a] for w, a in zip(omega, a2f) if w > 0]
    for w, a in wa:
        w *= 0.5 # Convert Ry to Ha
        a2fin.write("{0} {1}\n".format(w,a))
    a2fin.close()

    # Create elk input file
    elkin = open("tmp_elk/elk.in", "w")
    elkin.write("tasks\n260\n\nntemp\n20\nmustar\n{0}\n".format(mu))
    elkin.write("\nwplot\n{0} {1} {2}\n-0.5 0.5\n".format(len(wa), 1, 1))
    elkin.write("sppath\n'{0}'\n".format(species_dir))
    elkin.write("atoms\n1\n'La.in'\n1\n0 0 0 0 0 0\n")
    elkin.write("avec\n1 0 0\n0 1 0\n0 0 1")
    elkin.close()

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

    omegas = [w for w, a in wa]
    lam    = np.trapz([2*a/w for w, a in wa], x=omegas)
    wlog   = np.exp((2/lam)*np.trapz([np.log(w)*a/w for w, a in wa], x=omegas))
    wrms   = ((2/lam)*np.trapz([a*w for w, a in wa], x=omegas))**0.5

    g1 = 2.46*(1+3.8*mu)
    g2 = 1.82*(1+6.3*mu)*(wrms/wlog)

    f1 = (1+(lam/g1)**(3.0/2.0))**(1.0/3.0)
    f2 = 1 + (wrms/wlog - 1) * (lam**2) / (lam**2 + g2**2) 

    tc_ad = RY_TO_K*f1*f2*(wlog/1.20)*np.exp(-1.04*(1+lam)/(lam-mu-0.62*lam*mu))
    
    # Guess tc from where gaps reach < 5% of maximum
    tc_guess = 0
    dtc_guess = np.inf
    for i, (t, g) in enumerate(zip(ts, gaps)):
        if g < max(gaps)*0.05:
            tc_guess = t
            if i > 0:
                # Move slightly to the left as we could
                # have hit zero gap earlier (also helps
                # the curve fit to not get stuck)
                tc_guess = ts[i]*0.8 + ts[i-1]*0.2
            break

    # Fit to model to extract Tc from gap equations
    p0 = [tc_guess, max(gaps)] # Initial param guess from A-D
    try:
        par, cov = curve_fit(gap_model, ts, gaps, p0)
    except:
        par = [0]
        cov = np.inf

    if plot_fit:
        import matplotlib.pyplot as plt
        plt.plot(ts, gaps, label="Gap", marker="+")
        plt.plot(ts, gap_model(ts,*par), label="Fit $T_C={0}$".format(par[0]), linestyle=":", marker="+")
        plt.axvline(tc_guess, label="Guess", color="black", alpha=0.5)
        plt.legend()
        plt.show()

    if np.isfinite(cov).all(): 
        tc  = par[0]
        err = cov[0][0]**0.5
    else:
        tc  = tc_guess
        err = 0

    # See what elk thinks the allen-dynes parameters are
    with open("tmp_elk/ELIASHBERG.OUT") as f:
        for l in f.read().split("\n"):
            if "Mcmillan Tc"     in l: elk_ad   = float(l.split(":")[-1])
            if "Mcmillan lambda" in l: elk_lam  = float(l.split(":")[-1])
            if "Mcmillan wlog"   in l: elk_wlog = float(l.split(":")[-1])
            if "Mcmillan wrms"   in l: elk_wrms = float(l.split(":")[-1])

    print("Tc = {0} +/- {1} (Eliashberg)".format(tc, err))
    print("Mcmillan params")
    print("           Me        Elk")
    print(" Tc        {0:8.8} {1:8.8}  (ratio = {2})".format(tc_ad, elk_ad, tc_ad/elk_ad))
    print(" Lambda    {0:8.8} {1:8.8}  (ratio = {2})".format(lam,   elk_lam, lam/elk_lam))
    print(" Wlog      {0:8.8} {1:8.8}  (ratio = {2})".format(wlog,  elk_wlog, wlog/elk_wlog))
    print(" Wrms      {0:8.8} {1:8.8}  (ratio = {2})".format(wrms,  elk_wrms, wrms/elk_wrms))
    print(" Wlog/Wrms {0:8.8} {1:8.8}  (ratio = {2})".format(wlog/wrms, elk_wlog/elk_wrms,
         (wlog/wrms)/(elk_wlog/elk_wrms) ))
    print("\n")

    # Remove temporary directory
    #os.system("rm -r tmp_elk")

    return [tc, lam, wlog, tc_ad]

# List all files in the given folder
def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            yield os.path.join(root, filename)

# Find all a2F.dos* files in base_dir
# or its subdirectories and create a
# matching a2f.dos*.tc file containing tc
# info
def process_all_a2f(base_dir, overwrite=False, plot_fits=False, ignore_imaginary=False):

    for f in listfiles(base_dir):

        if not "a2f.dos" in f.lower(): continue
        if f.endswith(".tc"): continue
        print("\n")

        ftc = f+".tc"
        if os.path.isfile(ftc) and (not overwrite):
            print("Refusing to overwrite "+ftc)
            continue

        d = "/".join(f.split("/")[0:-1])

        print("Parsing a2F for "+f)
        omega, a2f, a2fnn, a2fp = parser.parse_a2f(f)
        if min(omega) < 0:
            if ignore_imaginary:
                print("Imaginary modes in {0}, ignoring them.".format(f))
            else:
                print("Could not get tc for {0}, imaginary modes detected.".format(f))
                continue

        print("Getting T_c for "+f)
        w = open(ftc,"w")

        tc, lam, wlog, tc_ad = get_tc_info(omega, a2f, 0.1, plot_fit=plot_fits)
        w.write("mu = 0.1\n")
        fs = "{0} # Tc (Eliashberg)\n{1} # Tc (Allen-Dynes)\n{2} # Lambda\n{3} # <w>\n"
        w.write(fs.format(tc,tc_ad,lam,wlog))

        tc, lam, wlog, tc_ad = get_tc_info(omega, a2f, 0.15, plot_fit=plot_fits)
        w.write("mu = 0.15\n")
        fs = "{0} # Tc (Eliashberg)\n{1} # Tc (Allen-Dynes)\n{2} # Lambda\n{3} # <w>\n"
        w.write(fs.format(tc,tc_ad,lam,wlog))

        w.close()
