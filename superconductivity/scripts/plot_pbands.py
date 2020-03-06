from quantum_espresso_tools.parser import parse_bands
from quantum_espresso_tools.plots  import plot_bands
import sys
import matplotlib.pyplot as plt

qs, ws = parse_bands(sys.argv[1])

hsp_file  = None
lifetimes = None

for i, arg in enumerate(sys.argv):
    if i < 2: continue
    if not arg.startswith("-"): continue

    if arg.startswith("-hsp"):
        hsp_file = sys.argv[i+1]

    if arg.startswith("-lifetimes"):
        qs_ignored, lifetimes = parse_bands(sys.argv[i+1])

plot_bands(qs, ws, "Frequency (cm^-1)", hsp_file=hsp_file, lifetimes=lifetimes)
plt.show()
