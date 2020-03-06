import sys
from quantum_espresso_tools.parser import parse_a2f
import matplotlib.pyplot as plt

RY_TO_CMM = 109736.75775046606

w, a2f, a2fnn, a2fp = parse_a2f(sys.argv[1])
plt.plot(a2f, w*RY_TO_CMM)
plt.axhline(0)
plt.show()
