import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=20)

with open("extract_evals.in","w") as f:
    f.write("&BANDS\n")
    f.write("outdir='.',\n")
    f.write("filband='extracted_evals.out',\n")
    f.write("lsym=.false.\n")
    f.write("/\n")

os.system("bands.x < extract_evals.in > extract_evals.out")

with open("scf.out") as f:
    for l in f.read().split("\n"):
        if "Fermi energy is" in l:
            fermi_energy = float(l.split()[-2])

with open("extracted_evals.out") as f:
    lines = f.read().split("\n")

evals = []
for l in lines:
    if len(l.split()) == 3:
        continue
    try:
        evals.extend([float(w) for w in l.split()])
    except:
        continue

RY_TO_EV = 13.605698065893753

data = []
for sigma in np.linspace(0.0001, 0.1, 101):
    sigma *= RY_TO_EV

    pts = []
    for e in evals:
        pts.append([e, np.exp(-(e-fermi_energy)**2/sigma**2)])
    pts = np.array(pts).T

    data.append([sigma/RY_TO_EV, np.sum(pts[1])/sigma])

data = np.array(data).T
plt.plot(data[0], data[1], marker="+")
plt.xlabel("$\sigma$ (Ry)")
plt.ylabel(r"$\frac{1}{\sigma} \sum_k \exp\left(\frac{(e_k - e_F)^2}{\sigma^2}  \right)$")
plt.show()
