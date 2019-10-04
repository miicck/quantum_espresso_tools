from quantum_espresso_tools.parser import parse_bands
from quantum_espresso_tools.plots  import plot_bands
import sys
import matplotlib.pyplot as plt

qs, ws = parse_bands(sys.argv[1])
hsp = None
if len(sys.argv) > 2: hsp = sys.argv[2]
plot_bands(qs, ws, "Frequency (cm^-1)", hsp_file=hsp)
plt.show()
