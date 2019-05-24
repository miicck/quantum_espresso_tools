import numpy as np
import seekpath
import sys
import os

# Parse the result of a vc-relax run for the
# atomic positions and the cell parameters
def parse_vc_relax(filename):
	
	f     = open(filename)
        lines = f.read().split("\n")
	f.close()

        start   = False
        lattice = []
        atoms   = []

        for i, line in enumerate(lines):
		
		# Ignore case, check if we're at the final coords output
                line = line.lower()
                if "begin final coordinates" in line: start = True
                if not start: continue

		# Parse cell parameters 
                if "cell_parameters" in line:
                        for j in range(i+1,i+4):
                                lattice.append([float(w) for w in lines[j].split()])

		# Parse atomic positions
                if "atomic_positions" in line:
                        for j in range(i+1, len(lines)):
                                try:
                                        name, x, y, z = lines[j].split()
                                        x, y, z = [float(c) for c in [x,y,z]]
                                        atoms.append([name, x, y, z])
                                except:
                                        break

		# We've got the atoms, so we're done
                if len(atoms) > 0:
                        break

        return [lattice, atoms]

# Get the seekpath representation of the primitive geometry
# for the given input file returns [atom_names, seekpath_geom]
def get_primitive(infile, cart_tol=0.01, angle_tol=5):

        fin = open(infile)
        lines = fin.read().split("\n")
        fin.close()

        lattice      = []
        frac_coords  = []
        atom_names   = []
        atom_numbers = []
        i_ignored    = []

        for i, line in enumerate(lines):
                if i in i_ignored: continue

                if "cell_parameters" in line.lower():
                        for j in range(i+1,i+4):
                                i_ignored.append(j)
                                lattice.append([float(w) for w in lines[j].split()])

                if "atomic_positions" in line.lower():
                        if "crystal" not in line.lower():
                                print "Only (crystal) coordinates supported!"
                        for j in range(i+1, len(lines)):
                                try:
                                        name,x,y,z = lines[j].split()
                                        frac_coords.append([float(x), float(y), float(z)])
                                        if name in atom_names:
                                                atom_numbers.append(atom_names.index(name))
                                        else:
                                                atom_names.append(name)
                                                atom_numbers.append(len(atom_names)-1)
                                except:
                                        break

        structure = (lattice, frac_coords, atom_numbers)
        return  [atom_names, seekpath.get_path(
                structure,
                with_time_reversal=True,
                symprec=cart_tol,
                angle_tolerance=angle_tol,
                threshold=0)]

# Set the geometry in the given input file from the given lattice
# and atoms in the format [[name, x, y, z], [name, x, y, z] ... ]
# also sets the cutoff, kpoint sampling and pressure (if present)
def modify_input(in_file,
	lattice  = None,
	atoms    = None,
	kpoints  = None,
	cutoff   = None,
	pressure = None,
	smearing = None,
	qpoints  = None,
	den_cutoff = None):

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
				continue

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

		# Replace qpoints in el-ph coupling
		if qpoints != None:
			if "nq1" in line.lower():
				line = "nq1={0},".format(qpoints[0])
			if "nq2" in line.lower():
				line = "nq2={0},".format(qpoints[1])
			if "nq3" in line.lower():
				line = "nq3={0},".format(qpoints[2])
			
		overwrite.write(line+"\n")

	overwrite.close()

# Get the path in the B.Z, interpolated to roughly
# num_points points
def get_bz_path(prim_geom, num_points):
	interp_path = []
	names = {}

	pairs = prim_geom["path"]
	for i, ab in enumerate(pairs):
		c1 = np.array(prim_geom["point_coords"][ab[0]])
		c2 = np.array(prim_geom["point_coords"][ab[1]])
		fs = "{0:10.10} {1:20.20} {2:5.5} {3:10.10} {4:20.20}"

		interp_path.append(c1)
		names[len(interp_path)-1] = ab[0]
		max_j = num_points/len(pairs)
		for j in range(1, max_j):
			fj = j/float(max_j)
			interp_path.append(c1+fj*(c2-c1))

		# Dont duplicate endpoints
		if i < len(pairs) - 1:
			if ab[1] == pairs[i+1][0]:
				continue

		interp_path.append(c2)
		names[len(interp_path)-1] = ab[1]

	return [names, interp_path]
