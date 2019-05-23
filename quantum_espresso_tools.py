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
	den_cutoff = None):

        input = open(in_file)
        lines = input.read().split("\n")
        input.close()

        overwrite = open(in_file, "w")
        i_ignored = []

        for i, line in enumerate(lines):
                if i in i_ignored: continue

		# Replace cell parameters
		if lattice != None:
			if "cell_parameters" in line.lower():
				for j in range(i+1, i+4):
					i_ignored.append(i)
					overwrite.write(" ".join(lattice[j-i-1])+"\n")
				continue

		# Replace atomic positions
		if atoms != None:
			if "atomic_positions" in line.lower():
				for j in range(i+1, len(lines)):
					try:
						name,x,y,z = lines[j].split()
						i_ignored.append(j)
					except:
						break
				for a in atoms:
					overwrite.write(" ".join(a)+"\n")
				continue

		# Replace the kpoint grid
		if kpoints != None:
			if "k_points" in line.lower():
				i_ignored.append(i+1)
				overwrite.write("K_POINTS automatic\n")
				overwrite.write(kpoints+"\n")
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
			
		overwrite.write(line+"\n")

	overwrite.close()
