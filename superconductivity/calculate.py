from quantum_espresso_tools.symmetry import get_kpoint_grid
from quantum_espresso_tools.parser   import parse_vc_relax
import numpy as np
import seekpath
import os

def default_parameters():
        return {
        "nodes"          : 1,        # Number of compute nodes to use
        "cores_per_node" : 1,        # Number of cores per compute node
        "mpirun"         : "mpirun", # Mpi caller (i.e mpirun or aprun)
        "pressure"       : 0,        # Pressure in GPa
        "press_conv_thr" : 0.05,     # Pressure convergence threshold
        "ecutwfc"        : 40,       # Plane wave cutoff in Ry
        "ecutrho"        : 400,      # Density cutoff in Ry
        "kpoint_spacing" : 0.05,     # Kpoint spacing in A^-1
        "kpts_per_qpt"   : 8,        # Ratio of N_ki to N_qi in grids
        "qpt_dense_mult" : 10,       # Ratio of dense qpt grid to coarse
        "ph_ndos"        : 1000,     # Number of points at which to calculate phonon DOS
        "elec_band_kpts" : 100,      # Points along the bandstructure
        "pseudo_dir"     : "./",     # Pseudopotential directory
        "forc_conv_thr"  : 1e-3,     # Force convergence threshold
        "degauss"        : 0.02,     # Smearing width in Ry
        "mixing_beta"    : 0.7,      # Electronic mixing
        "conv_thr"       : 1e-6,     # Electron energy convergence thresh (Ry)
        "symm_tol_cart"  : 0.01,     # Symmetry tolerance for frac coords
        "symm_tol_angle" : 5,        # Symmetry tolerance for angles (degrees)
        "elph_nsig"      : 10,       # Number of smearing witdths to use
        "elph_dsig"      : 0.005,    # Spacing of smearing widths (Ry)
        "lattice"        : 2.15*np.identity(3),               # Crystal lattice in angstrom
        "species"        : [["Li", 7.0, "Li.UPF"]],           # Species of atom/mass/pseudo
        "atoms"          : [["Li",0,0,0],["Li",0.5,0.5,0.5]], # Atom names and x,y,z coords
        }

def read_parameters(filename):
        
        # Get the parameters specified in the given file
        ret = default_parameters()

        f = open(filename)
        lines = f.read().split("\n")
        f.close()
        
        i_ignored = []
        string_args = ["pseudo_dir"]
        int_args    = ["nodes", "cores_per_node"]

        for i, l in enumerate(lines):
                
                l = l.strip()
                if i in i_ignored: continue
                if l.startswith("#"): continue
                if len(l) == 0: continue
                key = l.split()[0]

                # Parse the lattice from the input file
                if key == "lattice":
                        lattice = []
                        for j in range(i+1, i+4):
                                i_ignored.append(j)
                                lattice.append([float(w) for w in lines[j].split()])
                        ret["lattice"] = lattice
                        continue

                # Parse the atoms from the input file
                elif key == "atoms":
                        count = int(l.split()[1])
                        atoms = []
                        for j in range(i+1, i+1+count):
                                i_ignored.append(j)
                                n,x,y,z = lines[j].split()
                                atoms.append([n, float(x), float(y), float(z)])
                        ret["atoms"] = atoms
                        continue

                # Parse atomic species from input file
                elif key == "species":
                        count = int(l.split()[1])
                        species = []
                        for j in range(i+1, i+1+count):
                                i_ignored.append(j)
                                n,m,p = lines[j].split()
                                species.append([n, float(m), p])
                        ret["species"] = species
                        continue
                                
                # Parse key-value pairs from the input file
                val = l.split()[1]

                # Convert type for certain keys
                if   key in string_args : val = str(val)
                elif key in int_args    : val = int(val)
                else                    : val = float(val)

                if key in ret: ret[key] = val
                else: print("Unkown key when parsing {1}: {0}".format(key, filename))

        return ret

def create_qe_input_geom(parameters):

        # Create the part of a q.e input file that
        # describes the geometry of the crystal

        # Cell parameters card
        t  = "CELL_PARAMETERS (angstrom)\n"
        t += "{0} {1} {2}\n".format(*parameters["lattice"][0])
        t += "{0} {1} {2}\n".format(*parameters["lattice"][1])
        t += "{0} {1} {2}\n".format(*parameters["lattice"][2])

        # Atomic species card
        t += "ATOMIC_SPECIES\n"
        for s in parameters["species"]:
                t += "{0} {1} {2}\n".format(*s)

        # Atomic positions card
        t += "ATOMIC_POSITIONS (crystal)\n"
        for a in parameters["atoms"]:
                t += "{0} {1} {2} {3}\n".format(*a)

        # K-points card
        t += "K_POINTS automatic\n"
        t += "{0} {1} {2} 0 0 0\n".format(*parameters["kpoint_grid"])

        return t

def create_relax_in(parameters):

        # Create a quantum espresso relax.in file with
        # the given parameters

        # Control namelist
        t  = "&CONTROL\n"
        t += "calculation='vc-relax',\n"
        t += "pseudo_dir='{0}',\n".format(parameters["pseudo_dir"])
        t += "outdir='.',\n"
        t += "forc_conv_thr={0},\n".format(parameters["forc_conv_thr"])
        t += "/\n"

        # System namelist
        t += "&SYSTEM\n"
        t += "ntyp={0},\n".format(len(parameters["species"]))
        t += "nat={0},\n".format(len(parameters["atoms"]))
        t += "ibrav=0,\n"
        t += "ecutwfc={0},\n".format(parameters["ecutwfc"])
        t += "ecutrho={0},\n".format(parameters["ecutrho"])
        t += "occupations='smearing',\n"
        t += "degauss={0},\n".format(parameters["degauss"])
        t += "smearing='mv',\n"
        t += "/\n"

        # Electrons namelist
        t += "&ELECTRONS\n"
        t += "mixing_beta={0},\n".format(parameters["mixing_beta"])
        t += "conv_thr={0},\n".format(parameters["conv_thr"])
        t += "/\n"

        # Ions namelist
        t += "&IONS\n"
        t += "ion_dynamics='bfgs',\n"
        t += "/\n"

        # Cell namelist
        t += "&CELL\n"
        t += "cell_dynamics='bfgs',\n"
        t += "press={0},\n".format(parameters["pressure"]*10)
        t += "press_conv_thr={0},\n".format(parameters["press_conv_thr"]*10)
        t += "/\n"

        t += create_qe_input_geom(parameters)

        # Write the file
        f = open("relax.in","w")
        f.write(t)
        f.close()

def create_scf_non_geom(parameters):
        
        # Control namelist
        t  = "&CONTROL\n"
        t += "calculation='scf',\n"
        t += "pseudo_dir='{0}',\n".format(parameters["pseudo_dir"])
        t += "outdir='.',\n"
        t += "tprnfor=.true.,\n"
        t += "tstress=.true.,\n"
        t += "/\n"

        # System namelist
        t += "&SYSTEM\n"
        t += "ntyp={0},\n".format(len(parameters["species"]))
        t += "nat={0},\n".format(len(parameters["atoms"]))
        t += "ibrav=0,\n"
        t += "ecutwfc={0},\n".format(parameters["ecutwfc"])
        t += "ecutrho={0},\n".format(parameters["ecutrho"])
        t += "occupations='smearing',\n"
        t += "degauss={0},\n".format(parameters["degauss"])
        t += "smearing='mv',\n"
        t += "la2F=.true.,\n"
        t += "/\n"

        # Electrons namelist
        t += "&ELECTRONS\n"
        t += "mixing_beta={0},\n".format(parameters["mixing_beta"])
        t += "conv_thr={0},\n".format(parameters["conv_thr"])
        t += "/\n"

        return t

def create_scf_in(parameters):
        
        # Create the non-geometry and geomety parts
        t  = create_scf_non_geom(parameters)
        t += create_qe_input_geom(parameters)

        # Write the file
        f = open("scf.in","w")
        f.write(t)
        f.close()

def get_bz_path(parameters):

        # Work out the path, assigning proportional numbers
        # of kpoints to path segments according to their length
        path   = parameters["bz_path"]
        hsp    = parameters["high_symm_points"]
        points = parameters["elec_band_kpts"]
        segs   = [(np.array(hsp[p[0]]), np.array(hsp[p[1]])) for p in path]
        seg_lengths = [np.linalg.norm(p[0]-p[1]) for p in segs]
        tot_length  = np.sum(seg_lengths)
        seg_counts  = [max(int(points*sl/tot_length),2) for sl in seg_lengths]

        kpoints = []
        for i, c in enumerate(seg_counts):
                pts = np.linspace(0.0, 1.0, c+1)[:-2]
                pts = [segs[i][0] + (segs[i][1] - segs[i][0])*p for p in pts]
                kpoints.extend(pts)

        return kpoints

def create_bands_in(parameters):
        
        # Create the bands input file the same
        # way as an scf file, then replace the kpoints
        # with the path
        t  = create_scf_non_geom(parameters)
        t += create_qe_input_geom(parameters)

        t = t[0:t.find("K_POINTS")]
        t += "K_POINTS (crystal)\n"
        kpoints = get_bz_path(parameters)

        # Format kpoints into output
        t += "{0}\n".format(len(kpoints))
        w  = 1.0/float(len(kpoints))
        for k in kpoints:
                t += "{0} {1} {2} {3}\n".format(k[0],k[1],k[2],w)

        # Write the file
        f = open("bands.in","w")
        f.write(t)
        f.close()

def create_elph_in(parameters):

        # Create the input for an electron-phonon
        # interaction calculation
        t  = "Calculate electron-phonon coefficients\n"
        t += "&INPUTPH\n"
        t += "tr2_ph=1.0d-12,\n"
        t += "fildvscf='elph_vscf',\n"
        t += "outdir='.',\n"
        t += "electron_phonon='interpolated',\n"
        t += "el_ph_sigma={0},\n".format(parameters["elph_dsig"])
        t += "el_ph_nsigma={0},\n".format(parameters["elph_nsig"])
        t += "trans=.true.,\n"
        t += "ldisp=.true.,\n"
        t += "nq1={0},\n".format(parameters["qpoint_grid"][0])
        t += "nq2={0},\n".format(parameters["qpoint_grid"][1])
        t += "nq3={0},\n".format(parameters["qpoint_grid"][2])

        # Check if calculation was already underway
        # if so, make this a continuation run
        if os.path.isfile("elph.out"):
                t += "recover=.true.,\n"

        t += "/\n"

        # Write the file
        f = open("elph.in","w")
        f.write(t)
        f.close()

def create_q2r_in(parameters):

        # Creates the input file for q2r.x
        t  = "&INPUT\n"
        t += "zasr='simple',\n"
        t += "fildyn='matdyn',\n"
        t += "flfrc='force_constants',\n"
        t += "la2F=.true.\n"
        t += "/\n"

        # Write the file
        f = open("q2r.in", "w")
        f.write(t)
        f.close()

def create_ph_bands_in(parameters):
        
        # Creates the input files for calculating
        # the phonon bandstructure
        t  = "&INPUT\n"
        t += "asr='simple',\n"
        t += "flfrc='force_constants',\n"
        t += "flfrq='ph_bands.freq',\n"
        t += "la2F=.true.,\n"
        t += "dos=.false.\n"
        t += "/\n"

        qpoints = get_bz_path(parameters)
        t += "{0}\n".format(len(qpoints))
        for q in qpoints:
                t += "{0} {1} {2}\n".format(*q)
        
        # Write the file
        f = open("ph_bands.in","w")
        f.write(t)
        f.close()

def create_ph_dos_in(parameters):
        
        # Create input file for calculating phonon density of states
        qgrid = parameters["qpoint_grid"] 
        scale = parameters["qpt_dense_mult"]

        t  = "&input\n"
        t += "asr='simple',\n"
        t += "flfrc='force_constants',\n"
        t += "flfrq='ph_dos.freq',\n"
        t += "la2F=.true.,\n"
        t += "dos=.true.\n"
        t += "fldos='phonon.dos',\n"
        t += "nk1={0},\n".format(qgrid[0]*scale)
        t += "nk2={0},\n".format(qgrid[1]*scale)
        t += "nk3={0},\n".format(qgrid[2]*scale)
        t += "ndos={0}\n".format(parameters["ph_ndos"])
        t += "/\n"

        # Write the file
        f = open("ph_dos.in", "w")
        f.write(t)
        f.close()

def create_bands_x_in(parameters):
        
        # Create input file for reordering of bands etc
        t  = "&BANDS\n"
        t += "outdir=.,\n"
        t += "filband=bands.x.bands\n"
        t += "/\n"

        # Write the file
        f = open("bands.x.in", "w")
        f.write(t)
        f.close()


def run_qe(exe, file_prefix, parameters):

        # Dont rerun if already done
        if os.path.isfile(file_prefix+".out"):
                f = open(file_prefix+".out")
                s = f.read()
                f.close()
                if "JOB DONE" in s: return

        # Run quantum espresso with specified parallelism
        mpirun = parameters["mpirun"]
        ppn    = parameters["cores_per_node"]
        nodes  = parameters["nodes"]
        np     = nodes * ppn

        # Setup mpi invokation
        if mpirun == "mpirun":
                mpirun = "mpirun -np {0}".format(np)
        elif mpirun == "aprun":
                mpirun = "aprun -n {0}".format(np)
        else:
                raise ValueError("Unkown mpirun = {0}".format(mpirun))
        
        # Invoke the program 
        nk = min(nodes*2, np)
        qe_flags = "-nk {0}".format(nk)
        cmd = "{0} {1} {2} <{3}.in> {3}.out".format(mpirun, exe, qe_flags, file_prefix) 
        os.system(cmd)

def reduce_to_primitive(parameters):
        
        # Return a modified parameter
        # set with the primitive geometry

        frac_coords  = []
        atom_numbers = []
        unique_names = []

        for a in parameters["atoms"]:
                if not a[0] in unique_names:
                        unique_names.append(a[0])

        for a in parameters["atoms"]:
                frac_coords.append(a[1:])
                atom_numbers.append(unique_names.index(a[0]))

        # Use seekpath to get primitive geometry
        structure = (parameters["lattice"], frac_coords, atom_numbers)
        prim_geom = seekpath.get_path(
                structure,
                with_time_reversal=True,
                symprec=parameters["symm_tol_cart"],
                angle_tolerance=parameters["symm_tol_angle"],
                threshold=0)

        # Convert back to our representation
        new_atoms = []
        for t, f in zip(prim_geom["primitive_types"],
                        prim_geom["primitive_positions"]):
                new_atoms.append([unique_names[t], f[0], f[1], f[2]])
        
        # Overwrite new structure
        parameters["lattice"] = prim_geom["primitive_lattice"]
        parameters["atoms"]   = new_atoms
        parameters["bz_path"] = prim_geom["path"]
        parameters["high_symm_points"] = prim_geom["point_coords"]

        # Work out kpoint grid from spacing
        kpoint_grid = get_kpoint_grid(
                parameters["lattice"], parameters["kpoint_spacing"])

        # Work out the qpoint grid so there is at least
        # 1/kpts_per_qpt qpoints per kpoint
        kpq = parameters["kpts_per_qpt"]
        k2q = lambda k : max(int(0.5 + float(k)/float(kpq)), 1)
        qpoint_grid = [k2q(k) for k in kpoint_grid]
        parameters["qpoint_grid"] = qpoint_grid

        # Ensure k-point grid is multiple of qpoint grid
        parameters["kpoint_grid"] = [q*kpq for q in qpoint_grid]

        # Return resulting new parameter set
        return parameters

def run(parameters):

        # Reduce to primitive description
        parameters = reduce_to_primitive(parameters)
        
        # Caclulate relaxed geometry
        create_relax_in(parameters)
        run_qe("pw.x", "relax", parameters)
        relax_data = parse_vc_relax("relax.out")
        parameters["lattice"] = relax_data["lattice"]
        parameters["atoms"]   = relax_data["atoms"]

        # Run SCF with the new geometry
        create_scf_in(parameters)
        run_qe("pw.x", "scf", parameters)

        # Run elec-phonon calculation
        create_elph_in(parameters)
        run_qe("ph.x", "elph", parameters)

        # Convert dynamcial matricies etc to real space
        create_q2r_in(parameters)
        run_qe("q2r.x", "q2r", parameters)

        # Caclulate the phonon bandstructure
        create_ph_bands_in(parameters)
        run_qe("matdyn.x", "ph_bands", parameters)

        # Caclulate the phonon density of states
        create_ph_dos_in(parameters)
        run_qe("matdyn.x", "ph_dos", parameters)

        # Caculate the electronic bandstructure
        create_bands_in(parameters)
        run_qe("pw.x", "bands", parameters)

        # Re-order bands and calc band-related things
        create_bands_x_in(parameters)
        run_qe("bands.x", "bands.x", parameters)

def calculate(infile):
    run(read_parameters(infile))
