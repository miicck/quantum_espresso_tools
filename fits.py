import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def birch_murnaghan(v, e0, v0, b0, b0p):
    vr = (v0/v)**(2.0/3.0)
    return e0 + (9.0*v0*b0/16.0)*(b0p*(vr-1.0)**3 + ((vr-1.0)**2)*(6.0-4.0*vr))

def fit_birch_murnaghan(vdata, edata, pdata=None, fallback=None):

    while len(vdata) < 4:
        # Not enough data points, linearly extrapolate
        print("To few data points for Birch-Murnagahn fit, linearly extrapolating...")
        vdata = np.array([*vdata, 2*vdata[-1]-vdata[-2]])
        edata = np.array([*edata, 2*edata[-1]-edata[-2]])

    p0 = [np.mean(edata), np.mean(vdata), 1.0, 1.0]
     
    try:
        par, cov = curve_fit(birch_murnaghan, vdata, edata, p0=p0)
        model = lambda v : birch_murnaghan(v, *par)
    except Exception as ex:
        if not fallback is None:
            from scipy.interpolate import CubicSpline
            try:
                ve = list(zip(vdata, edata))
                ve.sort()
                vdata, edata = np.array(ve).T
                return [CubicSpline(vdata, edata), None, None]
            except Exception as exf:
                print("Could not fit fallback to")
                print(vdata)
                print(edata)
                print(pdata)
                raise exf
        print("Failed to fit birch murnaghan with plotted initial guess")
        print("Exception: ", ex)
        plt.plot(vdata, edata, label="Data", marker="+")
        plt.plot(vdata, birch_murnaghan(vdata, *p0), label="Guess", marker="+")
        plt.legend()
        plt.show()
        raise ex

    return [model, par, cov]
