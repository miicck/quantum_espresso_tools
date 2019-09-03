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

RY_TO_EV  = 13.605698065893753
evals     = np.array(evals) - fermi_energy
evals_sq  = evals**2
evals_abs = abs(evals)
evals_abs_sorted = np.array(sorted(evals_abs))

data = []
for sigma in np.linspace(0.0001, 0.1, 401):
    sigma *= RY_TO_EV
    to_sum = np.exp(-evals_sq/sigma**2)
    data.append([sigma/RY_TO_EV, np.sum(to_sum)/sigma])

data = np.array(data).T

# Find the flat bit nearish the mean
s = data[1]
means = np.mean(s)
stds  = np.std(s)
near_mean = lambda y : abs(y - means)/stds < 4

ddata = [[data[0][i], abs(s[i] - s[i-1])] for i in range(1, len(s)) if near_mean(s[i])]
ddata = np.array(ddata).T
min_i = list(ddata[1]).index(min(ddata[1]))
sig_best = ddata[0][min_i]

plt.subplot(311)
plt.plot(evals_abs_sorted/RY_TO_EV, marker="+")

plt.subplot(312)
plt.plot(data[0], data[1], marker="+")
plt.axvline(sig_best)
plt.axvline(evals_abs_sorted[40]/RY_TO_EV)
plt.axvline(evals_abs_sorted[2]/RY_TO_EV)
plt.xlabel("$\sigma$ (Ry)")
plt.ylabel(r"$\frac{1}{\sigma} \sum_k \exp\left(\frac{(e_k - e_F)^2}{\sigma^2}  \right)$")

plt.subplot(313)
plt.plot(ddata[0], ddata[1], marker="+")
plt.axvline(sig_best)
plt.xlabel("$\sigma$ (Ry)")
plt.ylabel(r"Change in $\frac{1}{\sigma} \sum_k \exp\left(\frac{(e_k - e_F)^2}{\sigma^2}  \right)$")

plt.show()
