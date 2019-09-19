import sys
from quantum_espresso_tools.superconductivity.plots import plot_tc_vs_p
import matplotlib.pyplot as plt

pu = False
if "-plot_unstable" in sys.argv:
    pu = True
    sys.argv.remove("-plot_unstable")

pdd = False
if "-plot_double_delta" in sys.argv:
    pdd = True
    sys.argv.remove("-plot_double_delta")
    

for d in sys.argv[1:]:
    plot_tc_vs_p(d, show=False, plot_unstable=pu, plot_double_delta_info=pdd)
plt.show()
