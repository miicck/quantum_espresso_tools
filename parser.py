import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sys
import os

RY_TO_K = 157887.6633481157

# Parse the result of a vc-relax run for the
# atomic positions and the cell parameters
def parse_vc_relax(filename):
        
        f     = open(filename)
        lines = f.read().split("\n")
        f.close()

        start = False
        data  = {}

        for i, line in enumerate(lines):
                
                # Ignore case
                line = line.lower()

                # Parse cell parameters 
                if "cell_parameters" in line:
                        data["lattice"] = []
                        for j in range(i+1,i+4):
                                data["lattice"].append([float(w) for w in lines[j].split()])

                # Parse atomic positions
                if "atomic_positions" in line:
                        data["atoms"] = []
                        for j in range(i+1, len(lines)):
                                try:
                                        name, x, y, z = lines[j].split()
                                        x, y, z = [float(c) for c in [x,y,z]]
                                        data["atoms"].append([name, x, y, z])
                                except:
                                        break

                # Parse final enthalpy
                if "final enthalpy" in line:
                        data["enthalpy"] = float(line.split("=")[-1].split("r")[0])

                # Parse final pressure
                if "p=" in line:
                        data["pressure"] = float(line.split("=")[-1])

                # Parse final volume
                if "unit-cell volume" in line:
                        data["volume"] = float(line.split("=")[-1].split()[0])
        
        return data

# Parse an scf.out file for various things
def parse_scf_out(filename):

        f = open(filename)
        lines = f.read().split("\n")
        f.close()
        
        data = {}

        for i, line in enumerate(lines):
                
                if "Fermi energy" in line:
                        data["fermi_energy"] = float(line.split("is")[-1].split("e")[0])
        return data

# Set the geometry in the given input file from the given lattice
# and atoms in the format [[name, x, y, z], [name, x, y, z] ... ]
# also sets the cutoff, kpoint sampling and pressure (if present)
def modify_input(in_file,
        lattice     = None,
        atoms       = None,
        kpoints     = None,
        cutoff      = None,
        pressure    = None,
        smearing    = None,
        qpoints     = None,
        calculation = None,
        den_cutoff  = None,
        recover     = None):

        input = open(in_file)
        lines = input.read().split("\n")
        input.close()

        overwrite = open(in_file, "w")
        i_ignored = []

        for i, line in enumerate(lines):
                if i in i_ignored: continue

                # Replace cell parameters
                if not lattice is None:
                        if "cell_parameters" in line.lower():
                                overwrite.write(line+"\n")
                                for j in range(i+1, i+4):
                                        i_ignored.append(j)
                                        overwrite.write(" ".join([str(x) for x in lattice[j-i-1]])+"\n")
                                continue

                # Replace atomic positions
                if not atoms is None:
                        if "atomic_positions" in line.lower():
                                for j in range(i+1, len(lines)):
                                        try:
                                                name,x,y,z = lines[j].split()
                                                i_ignored.append(j)
                                        except:
                                                break
                                overwrite.write(line+"\n")
                                for a in atoms:
                                        overwrite.write(" ".join([str(ai) for ai in a])+"\n")
                                continue

                # Replace the kpoint grid
                if kpoints != None:
                        if "k_points" in line.lower():
                                i_ignored.append(i+1)
                                if len(kpoints) == 3:
                                        overwrite.write("K_POINTS automatic\n")
                                        overwrite.write(" ".join([str(k) for k in kpoints])+" 0 0 0\n")
                                else:
                                        overwrite.write("K_POINTS (crystal)\n")
                                        overwrite.write(str(len(kpoints))+"\n")
                                        weight = 1/float(len(kpoints))
                                        for k in kpoints:
                                                kline = " ".join(str(ki) for ki in k)
                                                kline += " " + str(weight)
                                                overwrite.write(kline+"\n")
                                kpoints = None
                                continue

                # Replace number of atoms
                if atoms != None:
                        if "nat=" in line.replace(" ","").lower():
                                line = "nat="+str(len(atoms))+","
                        if "ntyp" in line.lower():
                                unique_names = []
                                for a in atoms:
                                        if a[0] in unique_names:
                                                continue
                                        unique_names.append(a[0])
                                line = "ntyp="+str(len(unique_names))+","

                # Replace the calculation type
                if calculation != None:
                        if "calculation" in line.lower():
                                line = "calculation="+calculation+","

                # Replace qpoints in el-ph coupling
                if qpoints != None:
                        if "nq1" in line.lower():
                                line = "nq1={0},".format(qpoints[0])
                        if "nq2" in line.lower():
                                line = "nq2={0},".format(qpoints[1])
                        if "nq3" in line.lower():
                                line = "nq3={0},".format(qpoints[2])

                # Replace the cutoff
                if cutoff != None:
                        if "ecutwfc" in line.lower(): 
                                line = "ecutwfc="+str(cutoff)+","

                # Replace the density cutoff
                if den_cutoff != None:
                        if "ecutrho" in line.lower():
                                line = "ecutrho="+str(den_cutoff)+","

                # Replace the electronic smearing amount (degauss)
                if smearing != None:
                        if "degauss" in line.lower():
                                line = "degauss="+str(smearing)+","

                # Replace the pressure in a relax file
                if pressure != None:
                        if "press" in line.lower():
                                line = "press="+str(pressure)+","

                # Replace the recovery option for a phonon calculation
                if recover != None:
                        if "recover" in line.lower():
                                if recover:
                                        line = "recover=.true.,"
                                else:
                                        line = "recover=.false.,"
                                # record our success
                                recover = None
                        
                overwrite.write(line+"\n")

        # Add kpoints to bottom of file
        # if they were not set somewhere else
        if kpoints != None:
                if len(kpoints) == 3:
                        overwrite.write("K_POINTS automatic\n")
                        overwrite.write(" ".join([str(k) for k in kpoints])+" 0 0 0\n")
                else:
                        overwrite.write("K_POINTS (crystal)\n")
                        overwrite.write(str(len(kpoints))+"\n")
                        weight = 1/float(len(kpoints))
                        for k in kpoints:
                                kline = " ".join(str(ki) for ki in k)
                                kline += " " + str(weight)
                                overwrite.write(kline+"\n")

        overwrite.close()
        
        # Check for errors
        if recover != None:
                ex  = "Did not properly set the recover option! "
                ex += "Does the line recover=... exist in "+in_file
                raise Exception(ex)

# Parse the eliashberg function from a given output file
def parse_a2f(a2f_file):
        
        data = []
        for line in open(a2f_file).read().split("\n"):
                if "lambda" in line:
                        lam = line.split("lambda")[-1]
                        lam = float(lam.split("=")[1].split()[0])
                if "." not in line: continue
                try:
                        dat = [float(n) for n in line.split()]
                except:
                        continue
                if len(dat) == 0: continue
                data.append(dat)

        data = np.array(data).T

        omega     = data[0]
        a2f_full  = data[1]
        a2f_proj  = data[2:]
        a2f_noneg = np.zeros(len(a2f_full))
        
        for p in a2f_proj:
                neg_mode = False
                for i in range(0, len(p)):
                        if omega[i] > 0: continue
                        if abs(p[i]) < 10e-4: continue
                        neg_mode = True
                        break
                if neg_mode: continue
                a2f_noneg += p
        
        return [omega, a2f_full, a2f_noneg, a2f_proj]

# Parse a .bands file
def parse_bands(bands_file):
        data = open(bands_file).read()

        # Parse first line for band_count, q_count then remove it
        lines      = data.split("\n")
        band_count = int(lines[0].split("=")[1].split(",")[0])
        q_count    = int(lines[0].split("=")[2].split("/")[0])
        data       = "\n".join(lines[1:])
        data = data.replace("-"," -")

        # Parse data into a list of q-points (qs) and a list of
        # frequencies for each (all_ws)
        # such that all_ws[i] corresponds to frequencies at qs[i]
        q  = []
        qs = []
        q_ws   = []
        all_ws = []
        cycle_count = 0

        for w in data.split():

                if cycle_count < 3:
                        q.append(float(w))
                        if len(q) == 3:
                                qs.append(q)
                                q = []

                else:
                        omega = float(w)
                        q_ws.append(omega)
                        if len(q_ws) == band_count:
                                all_ws.append(q_ws)
                                q_ws = []

                cycle_count += 1
                cycle_count = cycle_count % (3 + band_count)
        
        return [qs, all_ws]

# Parse partial electronic PDOS from all pdos_atom#... files
def parse_electron_pdos(direc):
        files = []
        for f in os.listdir(direc):
                if "pdos_atm" not in f: continue
                files.append(direc+"/"+f)

        pdos = []
        energies = []
        labels = []
        read_energies = False   

        for f in files:
                atm, wf = f.split("_")[-2:]
                atm = "Atom "+atm.split("#")[-1]
                wf = wf.split("#")[-1]
                labels.append(atm + " " + wf)
                
                f = open(f)
                lines = f.read().split("\n")[1:-1]
                f.close()
        
                data = []
                for l in lines:
                        data.append(float(l.split()[2]))
                        if not read_energies:
                                energies.append(float(l.split()[0]))
                pdos.append(data)
                read_energies = True
        
        return energies, pdos, labels

# Parse phonon density of states from phonon.dos file
def parse_phonon_dos(filename):
        f = open(filename)
        lines = f.read().split("\n")[1:-1]
        f.close()
        data = []
        for l in lines:
                data.append([float(w) for w in l.split()])
        data = np.array(data).T
        return data[0], data[2:] # Note: data[1] = sum(data[2:])

# Plot a density of states, (or partial density of states)
def plot_dos(ws, pdos, labels=None, fermi_energy=0):
        tot = np.zeros(len(pdos[0]))
        ws = np.array(ws) - fermi_energy
        for i, pd in enumerate(pdos):
                label = None
                if not labels is None:
                        label = labels[i]
                plt.fill_betweenx(ws, tot, tot+pd, label=label)
                tot += pd
        if not labels is None:
                plt.legend()
        plt.axhline(0, color="black")

# Plot a bandstructure (optionally specifying a file
# with the indicies of the high symmetry points)
def plot_bands(qs, all_ws, ylabel, hsp_file=None, fermi_energy=0, resolve_band_cross=False):

        # Parse high symmetry points
        if hsp_file is None:
                xtick_vals  = [0, len(qs)]
                xtick_names = ["", ""] 
        else:
                lines = open(hsp_file).read().split("\n")
                xtick_vals = []
                xtick_names = []
                for l in lines:
                        if len(l.split()) == 0: continue
                        index, name = l.split()
                        xtick_vals.append(int(index))
                        xtick_names.append(name)

        # Find discontinuities in the path
        dc_pts = []
        for i in xtick_vals:
                for j in xtick_vals:
                        if j >= i: continue
                        if abs(i-j) == 1:
                                dc_pts.append(i)

        # Attempt to sort out band crossings
        bands = np.array(all_ws)
        for iq in range(1, len(bands) - 1):
                if not resolve_band_cross: break

                # Extrapolate modes at iq+1 from modes at iq and iq-1
                extrap = []
                for im in range(0, len(bands[iq])):
                        extrap.append(2*bands[iq][im]-bands[iq-1][im])

                # Swap iq+1'th bands around until they minimize
                # difference to extrapolated values
                swap_made = True
                while swap_made:

                        swap_made = False
                        for it in range(1, len(bands[iq])):

                                # Dont swap bands which are of equal value
                                if (bands[iq+1][it] == bands[iq+1][it-1]): continue

                                # If the order of extrapolated bands at iq+1 is not equal
                                # to the order of bands at iq+1, swap the bands after iq
                                if (extrap[it] < extrap[it-1]) != (bands[iq+1][it] < bands[iq+1][it-1]):

                                        for iqs in range(iq+1, len(bands)):
                                                tmp = bands[iqs][it]
                                                bands[iqs][it]   = bands[iqs][it-1]
                                                bands[iqs][it-1] = tmp

                                        swap_made = True
        bands = bands.T

        # Plot the bands between each successive pair
        # of discontinuities
        dc_pts.append(0)
        dc_pts.append(len(bands[0]))
        dc_pts.sort()
        for band in bands:
                for i in range(1, len(dc_pts)):
                        s = dc_pts[i-1]
                        f = dc_pts[i]
                        plt.plot(range(s,f),band[s:f]-fermi_energy,color=np.random.rand(3))

        plt.axhline(0, color="black")
        plt.ylabel(ylabel)
        plt.xticks(xtick_vals, xtick_names)
        for x in xtick_vals:
                plt.axvline(x, color="black", linestyle=":")

        for x in dc_pts:
                plt.axvline(x, color="black")
                plt.axvline(x-1, color="black")

# Removes the brillouin zone path from a bandstructure input file
def remove_bz_path(bands_in):
        read  = open(bands_in)
        lines = read.read().split("\n")
        read.close()
        write = open(bands_in,"w")
        for l in lines:
                if "/" in l: 
                        write.write("/\n")
                        break 
                write.write(l+"\n")
        write.close()
