import os
from subprocess import check_output
from quantum_espresso_tools import parser
from scipy.optimize import curve_fit
import numpy as np
import warnings
warnings.filterwarnings("error")

RY_TO_K = 157887.6633481157

# Model of the superconducting gap vs temperature
# used to fit for Tc
def gap_model(t, tc, gmax):
    t = np.array([min(ti, tc) for ti in t])
    return gmax * np.tanh(1.74*np.sqrt(tc/t - 1))

# Get superconductivity info from eliashhberg function
# we ignore the portion of a2f where w < 0 (if there is such a region)
def get_tc_info(omega, a2f, mu, plot_fit=False, plot_errors=False, outf=None):

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
    if not outf is None:
        outf.write("Solving eliashberg equations with mu = {0} ...\n".format(mu))
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
        gaps.append(vals[1]/vals[2])

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
    if tc_guess < 1.0: tc_guess = tc_ad
    p0 = [tc_guess, max(gaps)] 
    try:
        par, cov = curve_fit(gap_model, ts, gaps, p0)

    except Warning as warn:
        if not outf is None:
            outf.write("Fit failed with waring:\n")
            outf.write(str(warn)+"\n")

        if plot_errors:
            import matplotlib.pyplot as plt
            plt.plot(ts, gaps)
            plt.axvline(tc_guess, label="Guess")
            plt.legend()
            plt.show()

        par = [0]
        cov = np.inf

    except Exception as err:
        outf.write("Fit failed with errror:\n")
        outf.write(err+"\n")

        if plot_errors:
            import matplotlib.pyplot as plt
            plt.plot(ts, gaps)
            plt.axvline(tc_guess, label="Guess")
            plt.legend()
            plt.show()

        raise err

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
        outf.write("Covariance is infinite!\n")
        tc  = tc_guess
        err = np.inf

    outf.write("Tc = {0} +/- {1} (Eliashberg)\n".format(tc, err))
    outf.write("Mcmillan params      \n")
    outf.write("    Tc        {0} K  \n".format(tc_ad))
    outf.write("    Lambda    {0}    \n".format(lam))
    outf.write("    Wlog      {0} Ry \n".format(wlog))
    outf.write("    Wrms      {0} Ry \n".format(wrms))
    outf.write("    Wlog/Wrms {0}    \n".format(wlog/wrms))

    # Remove temporary directory
    os.system("rm -r tmp_elk")

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
def process_all_a2f(base_dir, overwrite=False, plot_fits=False):

    # Open the output and error files
    outf = open(base_dir+"/postprocess_out","w",1) 
    errf = open(base_dir+"/postprocess_errors","w",1)

    # Loop over all files in base_dir or subdirectories
    for f in listfiles(base_dir):

        # Find a2f.dos{n} files
        if not "a2f.dos" in f.lower(): continue
        if f.endswith(".tc"): continue # Not already processed

        outf.write("\n")

        # Dont overwrite unless instructed to do so
        ftc = f+".tc"
        if os.path.isfile(ftc) and (not overwrite):
            outf.write("Refusing to overwrite "+ftc+"\n")
            continue

        # Find the directory that this a2f file is in
        d = "/".join(f.split("/")[0:-1])

        # Attempt to calculate Tc for this a2F
        try:
            # Parse a2F
            outf.write("Parsing a2F for "+f+"\n")
            omega, a2f, a2fnn, a2fp = parser.parse_a2f(f)
            dynamically_unstable = False
            for w, a in zip(omega, a2f):
                if w > -10e-10: continue
                if a <  10e-10: continue
                dynamically_unstable = True
                outf.write("Dynamically unstable\n")
                break

            # Solve eliashberg equations using elk for
            # mu* = 0.1 and mu* = 0.15
            outf.write("Getting T_c for "+f+"\n")
            w = open(ftc,"w")

            tc, lam, wlog, tc_ad = get_tc_info(omega, a2f, 0.1, 
                plot_fit=plot_fits, outf=outf)

            w.write("mu = 0.1\n")
            fs  = "{0} # Tc (Eliashberg)\n"
            fs += "{1} # Tc (Allen-Dynes)\n"
            fs += "{2} # Lambda\n{3} # <w>\n"
            w.write(fs.format(tc,tc_ad,lam,wlog))

            tc, lam, wlog, tc_ad = get_tc_info(omega, a2f, 0.15, 
                plot_fit=plot_fits, outf=outf)

            w.write("mu = 0.15\n")
            fs =  "{0} # Tc (Eliashberg)\n"
            fs += "{1} # Tc (Allen-Dynes)\n"
            fs += "{2} # Lambda\n{3} # <w>\n"
            w.write(fs.format(tc,tc_ad,lam,wlog))
            fs =  "{0} # Dynamically unstable?"
            fs += "If true, we have ignored imaginary modes to obtain Tc\n"
            w.write(fs.format(dynamically_unstable))

            w.close()

        except Exception as e:

            # Log errors
            errf.write("Error while processing {0}:\n".format(f))
            errf.write(str(e)+"\n")

    # Close output files
    outf.close()
    errf.close()
