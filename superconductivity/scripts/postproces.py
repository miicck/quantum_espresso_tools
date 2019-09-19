from quantum_espresso_tools.superconductivity.postprocess import process_all_a2f
import sys

overwrite   = "-f" in sys.argv
process_all_a2f(".", overwrite=overwrite)
