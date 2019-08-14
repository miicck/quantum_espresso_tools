import seekpath
import numpy as np

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
                                print("Only (crystal) coordinates supported!")
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

# Get a kpoint grid for a given lattice and spacing
# (I worked this out using inverse angstrom spacing and 
#  angstrom lattice, but it should generalize to any units)
def get_kpoint_grid(lattice, kpoint_spacing):

        recip_lattice = np.linalg.inv(lattice).T
        return [int(np.linalg.norm(b)/kpoint_spacing) for b in recip_lattice]
