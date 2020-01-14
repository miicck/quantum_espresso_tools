import numpy as np
import matplotlib.pyplot as plt
import os

pdos = {}

for f in os.listdir("."):

    if "pdos_atm" not in f: continue
    p, a, w = f.split("#")
    a = a.split("(")[-1].split(")")[0]

    if a not in pdos: pdos[a] = {}

    with open(f) as dosf:
        for line in dosf:
            if line.startswith("#"): continue
            try:
                e, l = [float(w) for w in line.split()][0:2]
                if e not in pdos[a]:
                    pdos[a][e] = 0
                pdos[a][e] += l
            except ValueError:
                continue

for a in pdos:
    pda = []
    print(a)
    for e in pdos[a]:
        pda.append([e, pdos[a][e]])

    es, dos = np.array(pda).T
    plt.plot(es, dos, label=a)

plt.legend()
plt.show()
