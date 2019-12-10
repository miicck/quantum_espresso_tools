import sys
from quantum_espresso_tools.superconductivity.calculate import calculate
dry = "-dry" in sys.argv

calculate(sys.argv[1], dry=dry)
