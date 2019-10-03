import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

# Convert Ry/a.u^3 to KBar 
RY_PER_AU3_TO_KBAR = 0.5 * 29421.02648438959 * 10

def bm_e(v, e0, v0, b0, b0p):
    vr = (v0/v)**(2.0/3.0)
    return e0 + (9.0*v0*b0/16.0)*(b0p*(vr-1.0)**3 + ((vr-1.0)**2)*(6.0-4.0*vr))

def bm_p(v, v0, b0, b0p):
    vr73 = (v0/v)**(7.0/3.0)
    vr53 = (v0/v)**(5.0/3.0)
    vr23 = (v0/v)**(2.0/3.0)
    return 1.5*b0*(vr73 - vr53)*(1 + 0.75*(b0p-4.0)*(vr23-1))

def rv_p(v, v0, b0, b0p):
    n = (v/v0)**(1.0/3.0)
    return 3*b0*((1-n)/(n**2.0))*np.exp(1.5*(b0p-1)*(1-n))

def mur_p(v, v0, k0, k0p):
    return (k0/k0p)*( ((v/v0)**(-k0p)) - 1 )

def mur_e(v, e0, v0, k0, k0p):
    vr = v/v0
    f  = 1/(k0p-1)
    return e0 + k0*v0*( f*(vr**(1-k0p))/k0p + vr/k0p - f )

def fit_birch_murnaghan(vdata, edata, p_guess=None):
    return fit_eos(vdata, edata, bm_e, bm_p, p_guess=p_guess)

def fit_eos(vdata, edata, eos_e, eos_p, p_guess=None):

    p0  = [np.mean(edata), np.mean(vdata), 1.0, 1.0]
    p0p = p0[1:]

    if not p_guess is None:

        try:
            # Try to fit EOS to guessed P(v) data, in order
            # to obtain a good initial guess for parameters
            p_fit     = np.array(p_guess)/RY_PER_AU3_TO_KBAR
            par, cov  = curve_fit(eos_p, vdata, p_fit, p0=p0p)
            p0[1:]    = par

        except Exception as ex:
            print("Could not EOS to plotted pressure data!")
            print(ex)
            plt.plot(vdata, p_fit)
            plt.plot(vdata, eos_p(vdata, *p0p))
            plt.show()

    try:
        # Try to fit EOS to the given E(v) data 
        par, cov = curve_fit(eos_e, vdata, edata, p0=p0)

        if not np.isfinite(cov).all():
            raise Exception("Infinite covariance detected in EOS fit!")

    except Exception as ex:
    
        try:
            print("Warning, switching to minmization instead of curve_fit!")
            print(ex)
            to_min = lambda p : np.linalg.norm(eos_e(vdata, *p) - edata)
            par = minimize(to_min, x0=p0).x

        except:
            print("Failed to fit EOS to plot shown!")
            plt.plot(vdata, edata, marker="+", label="data")
            plt.plot(vdata, eos_e(vdata, *p0), label="Guess")
            plt.legend()
            plt.show()
            raise ex

    # Construct the successful EOS model(s)
    e_model  = lambda v : eos_e(v, *par)
    p_model  = lambda v : RY_PER_AU3_TO_KBAR * eos_p(v, *par[1:])
    return [e_model, p_model, par, cov]
