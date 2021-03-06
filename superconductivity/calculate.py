from quantum_espresso_tools.symmetry import get_kpoint_grid
from quantum_espresso_tools.parser   import parse_vc_relax
import numpy as np
import numpy.linalg as la
import os
import subprocess

# Conversion factors
ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROM = 0.529177

def default_parameters():

    # Default pseudopotential directory is home/pseudopotentials
    if "HOME" in os.environ:
        pseudo_dir = os.environ["HOME"]+"/pseudopotentials"
    else:
        pseudo_dir = "./"

    # Try to get the number of cores from multiprocessing
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    except:
        cores = 1

    return {
    "nodes"            : 1,          # Number of compute nodes to use
    "cores_per_node"   : cores,      # Number of cores per compute node
    "mpirun"           : "mpirun",   # Mpi caller (i.e mpirun or aprun)
    "elph"             : True,       # True if we are to calculate electron-phonon coupling
    "relax_only"       : False,      # True if we are only to calculate relaxation
    "pressure"         : 0,          # Pressure in GPa
    "press_conv_thr"   : 0.5,        # Pressure convergence threshold
    "ecutwfc"          : 30,         # Plane wave cutoff in Ry
    "ecutrho"          : 300,        # Density cutoff in Ry
    "qpoint_spacing"   : 0.2,        # Q-point grid spacing in A^-1
    "kpts_per_qpt"     : [8, 8, 8],  # K-point grid (as multiple of q-point grid)
    "aux_kpts"         : None,       # Auxilliary k-point grid (as multiple of q-point grid)
    "kpoint_grid"      : None,       # Explicit primary k-point grid (overrides kpts_per_qpt, 
                                     # sets aux_kpts = None)
    "qpt_dense_mult"   : 10,         # Ratio of dense (interpolated) qpt grid to coarse q_point_grid
    "ph_ndos"          : 500,        # Number of points at which to calculate phonon DOS
    "band_kpts"        : 100,        # Points along the bandstructure
    "pseudo_dir"       : "./",       # Pseudopotential directory
    "forc_conv_thr"    : 1e-5,       # Force convergence threshold
    "degauss"          : 0.02,       # Smearing width in Ry
    "mixing_beta"      : 0.7,        # Electronic mixing
    "conv_thr"         : 1e-8,       # Electron energy convergence thresh (Ry)
    "symm_tol_cart"    : 0.001,      # Symmetry tolerance for frac coords
    "symm_tol_angle"   : 0.5,        # Symmetry tolerance for angles (degrees)
    "require_prim_geom": True,       # If true error will be thrown if we can't reduce to prim geom
    "elph_nsig"        : 10,         # Number of smearing witdths to use
    "elph_dsig"        : 0.01,       # Spacing of smearing widths (Ry)
    "disk_usage"       : "normal",   # Set to 'minimal' to delete unnessacary files
    "pseudo_dir"       : pseudo_dir, # Where the pseudopotentials for this run are
    "irrep_group_size" : 0,          # The number of irreps processed in each el-ph step (0 => all)
    "lattice"          : 2.15*np.identity(3),               # Crystal lattice in angstrom
    "species"          : [["Li", 7.0, "Li.UPF"]],           # Species of atom/mass/pseudo
    "atoms"            : [["Li",0,0,0],["Li",0.5,0.5,0.5]], # Atom names and x,y,z coords
    }

def read_parameters(filename):
    
    # Get the parameters specified in the given file
    ret = default_parameters()

    f = open(filename)
    lines = f.read().split("\n")
    f.close()
    
    i_ignored = []
    string_args = ["pseudo_dir", "disk_usage"]
    int_args    = ["nodes", "cores_per_node"]
    bool_args   = ["elph", "relax_only", "require_prim_geom"]

    for i, l in enumerate(lines):
        
        l = l.strip()
        if i in i_ignored:    continue # Skip lines that have been dealt with
        if l.startswith("#"): continue # Ignore comment lines
        if len(l) == 0:       continue # Ignore empty lines
        if l.find("#") >= 0:
            l = l[0:l.find("#")]  # strip comments
        key = l.split()[0]

        # Parse the lattice from the input file
        if key == "lattice":

            # Convert units to angstrom
            units = l.split()[1]
            if units == "angstrom":
                factor = 1.0
            elif units == "bohr":
                factor = BOHR_TO_ANGSTROM
            else:
                raise ValueError("Unkown lattice units: "+units)

            lattice = []
            for j in range(i+1, i+4):
                    i_ignored.append(j)
                    lattice.append([factor*float(w) for w in lines[j].split()])
            ret["lattice"] = lattice
            continue

        # Parse the atoms from the input file
        elif key == "atoms":
            count = int(l.split()[1])
            units = l.split()[2]

            # Calculate transformation to fractional coordinates
            # from given system of units. This is done by constructing
            # the matrix that transforms from the given units to 
            # crystal coords: the inverse lattice transpose linv.
            if units == "crystal" or units == "fractional":
                linv = np.identity(3)
            elif units == "angstrom":
                linv = la.inv(np.array(ret["lattice"]).T)
            elif units == "bohr":
                linv = la.inv(ANGSTROM_TO_BOHR*np.array(ret["lattice"]).T)
            else:
                raise ValueError("Unkown atom coordinate units: "+units)
            
            # Parse atoms, transforming as we go
            atoms = []
            for j in range(i+1, i+1+count):
                i_ignored.append(j)
                n,x,y,z = lines[j].split()
                r = [float(x), float(y), float(z)]
                try: r = np.matmul(linv, r)
                except AttributeError: r = np.dot(linv, r)
                atoms.append([n, r[0], r[1], r[2]])

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

        # Parse mpirun and arguments
        elif key == "mpirun":
            ret["mpirun"] = " ".join(l.split()[1:])
            continue

        # Parse qpoint_grid
        elif key == "qpoint_grid":
            ret["qpoint_grid"] = [int(q) for q in l.split()[1:4]]
            continue

        # Parse primary k-point grid
        elif key == "kpts_per_qpt":
            grid = [int(k) for k in l.split()[1:]]
            if   len(grid) == 3: ret[key] = grid
            elif len(grid) == 1: ret[key] = [grid[0], grid[0], grid[0]]
            else: raise Exception("Could not parse kpts_per_qpts from : "+l)
            continue

        # Parse aux k-point grid
        elif key == "aux_kpts":
            try:    
                grid = [int(k) for k in l.split()[1:4]]
                if   len(grid) == 3: ret[key] = grid
                elif len(grid) == 1: ret[key] = [grid[0], grid[0], grid[0]]
            except: 
                ret[key] = None # No auxillary grid
            continue

        # Parse explicit k-point grid
        elif key == "kpoint_grid":
            ret["kpoint_grid"] = [int(k) for k in l.split()[1:4]]
            continue
                        
        # Parse simple key-value pairs from the input file
        val = l.split()[1]

        # Convert type for certain keys
        if   key in string_args : val = str(val)
        elif key in int_args    : val = int(val)
        elif key in bool_args   : val = val.lower().strip() == "true"
        else                    : val = float(val) # Default to float

        if key in ret: ret[key] = val
        else: raise Exception("Unkown key when parsing {1}: {0}".format(key, filename))

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
    if os.path.isfile("relax.out"):
        t += "restart_mode='restart',\n"
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

def create_scf_non_geom(parameters, file_prefix="scf"):

    elph = parameters["elph"]
        
    # Control namelist
    t  = "&CONTROL\n"
    t += "calculation='scf',\n"
    t += "pseudo_dir='{0}',\n".format(parameters["pseudo_dir"])
    t += "outdir='.',\n"
    t += "tprnfor=.true.,\n"
    t += "tstress=.true.,\n"
    if os.path.isfile("{0}.out".format(file_prefix)):
        t += "restart_mode='restart',\n"
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
    if elph:
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

def create_extract_evals_in(parameters):
    
    # Create file to extract eigenvalues from scf run
    t  = "&BANDS\n"
    t += "outdir='.',\n"
    t += "filband='extracted_evals.out',\n"
    t += "lsym=.false.\n"
    t += "/\n"

    # Write the file
    f = open("extract_evals.in", "w")
    f.write(t)
    f.close()

def get_bz_path(parameters):

    # Work out the path, assigning proportional numbers
    # of kpoints to path segments according to their length
    path   = parameters["bz_path"]
    hsp    = parameters["high_symm_points"]
    points = parameters["band_kpts"]

    # Decide how long to make each segment
    segs        = [(np.array(hsp[p[0]]), np.array(hsp[p[1]])) for p in path]
    seg_lengths = [np.linalg.norm(p[0]-p[1]) for p in segs]
    tot_length  = np.sum(seg_lengths)
    seg_counts  = [max(int(points*sl/tot_length),2) for sl in seg_lengths]

    kpoints = []
    special_kpoints = []

    for i, c in enumerate(seg_counts):

        # Generate the points along this segment
        pts = np.linspace(0.0, 1.0, c+1)
        pts = [segs[i][0] + (segs[i][1] - segs[i][0])*p for p in pts]

        # Add them to the path, recording the start and
        # end indicies of the segment
        i_start = len(kpoints)
        kpoints.extend(pts)
        i_end   = len(kpoints)-1

        # Record the start and end point names
        special_kpoints.append([i_start, path[i][0], kpoints[i_start]])
        special_kpoints.append([i_end,   path[i][1], kpoints[i_end]])

    return [kpoints, special_kpoints]

def create_bands_in(parameters):
        
    # Create the bands input file the same
    # way as an scf file, then replace the kpoints
    # with the path
    t  = create_scf_non_geom(parameters, file_prefix="bands")
    t += create_qe_input_geom(parameters)

    t = t[0:t.find("K_POINTS")]
    t += "K_POINTS (crystal)\n"
    kpoints, special_kpoints = get_bz_path(parameters)

    # Write the high symmetry poitns to file
    with open("bands.high_symmetry_points","w") as f:
        for skp in special_kpoints:
            f.write("{0} {1} {2}\n".format(*skp))

    # Format kpoints into output
    t += "{0}\n".format(len(kpoints))
    w  = 1.0/float(len(kpoints))
    for k in kpoints:
        t += "{0} {1} {2} {3}\n".format(k[0],k[1],k[2],w)

    # Write the file
    with open("bands.in","w") as f:
        f.write(t)

def create_elph_in(
    name, 
    parameters, 
    q_range=None,
    irr_range=None,
    force_recover=False):

    elph = parameters["elph"]

    # Create the input for an electron-phonon
    # interaction calculation
    if elph: t = "Calculate electron-phonon coefficients\n"
    else:    t = "Calculate dynamical matrix\n"
    t += "&INPUTPH\n"
    t += "tr2_ph=1.0d-12,\n"
    t += "outdir='.',\n"
    t += "reduce_io=.true.,\n"
    t += "trans=.true.,\n"
    t += "ldisp=.true.,\n"
    t += "nq1={0},\n".format(parameters["qpoint_grid"][0])
    t += "nq2={0},\n".format(parameters["qpoint_grid"][1])
    t += "nq3={0},\n".format(parameters["qpoint_grid"][2])
    if not q_range is None:
        # Specify a range of q-points to calculate
        t += "start_q={0},\n".format(q_range[0])
        t += "last_q={0},\n".format(q_range[1])
    if not irr_range is None:
        # Specify a range of irreps to calculate
        t += "start_irr={0},\n".format(irr_range[0])
        t += "last_irr={0},\n".format(irr_range[1])
    if elph:
        # Parameters for electron-phonon calculation
        t += "fildvscf='elph_vscf',\n"
        t += "electron_phonon='interpolated',\n"
        t += "el_ph_sigma={0},\n".format(parameters["elph_dsig"])
        t += "el_ph_nsigma={0},\n".format(int(parameters["elph_nsig"]))

    # Check if calculation was already underway
    # if so, make this a continuation run
    if force_recover or os.path.isfile("elph.out"):
        t += "recover=.true.,\n"

    t += "/\n"

    # Write the file
    with open(name+".in","w") as f:
        f.write(t)

def create_q2r_in(parameters):

    elph = parameters["elph"]

    # Creates the input file for q2r.x
    t  = "&INPUT\n"
    t += "zasr='simple',\n"
    t += "fildyn='matdyn',\n"
    t += "flfrc='force_constants',\n"
    if elph:
        t += "la2F=.true.\n"
        t += "el_ph_nsigma={0},\n".format(int(parameters["elph_nsig"]))
    t += "/\n"

    # Write the file
    f = open("q2r.in", "w")
    f.write(t)
    f.close()

def create_ph_bands_in(parameters):

    elph = parameters["elph"]

    # Creates the input files for calculating
    # the phonon bandstructure
    t  = "&INPUT\n"
    t += "asr='simple',\n"
    t += "flfrc='force_constants',\n"
    t += "flfrq='ph_bands.freq',\n"
    t += "q_in_cryst_coord=.true.,\n"
    t += "dos=.false.\n"
    if elph:
        t += "la2F=.true.,\n"
        t += "el_ph_nsigma={0},\n".format(int(parameters["elph_nsig"]))
    t += "/\n"

    qpoints, special_qpoints = get_bz_path(parameters)

    with open("ph_bands.high_symmetry_points","w") as f:
        for sqp in special_qpoints:
            f.write("{0} {1} {2}\n".format(*sqp))

    t += "{0}\n".format(len(qpoints))
    for q in qpoints:
        t += "{0} {1} {2}\n".format(*q)
    
    # Write the file
    with open("ph_bands.in","w") as f:
        f.write(t)

def create_ph_dos_in(parameters):
        
    # Create input file for calculating phonon density of states
    qgrid = parameters["qpoint_grid"] 
    scale = parameters["qpt_dense_mult"]
    elph  = parameters["elph"]

    t  = "&input\n"
    t += "asr='simple',\n"
    t += "flfrc='force_constants',\n"
    t += "flfrq='ph_dos.freq',\n"
    t += "dos=.true.\n"
    t += "fldos='phonon.dos',\n"
    t += "nk1={0},\n".format(qgrid[0]*scale)
    t += "nk2={0},\n".format(qgrid[1]*scale)
    t += "nk3={0},\n".format(qgrid[2]*scale)
    t += "ndos={0}\n".format(int(parameters["ph_ndos"]))
    if elph:
        t += "la2F=.true.,\n"
        t += "el_ph_nsigma={0},\n".format(int(parameters["elph_nsig"]))
    t += "/\n"

    # Write the file
    with open("ph_dos.in", "w") as f:
        f.write(t)

def create_bands_x_in(parameters):
        
    # Create input file for reordering of bands etc
    t  = "&BANDS\n"
    t += "outdir='.',\n"
    t += "filband='bands.x.bands'\n"
    t += "/\n"

    # Write the file
    f = open("bands.x.in", "w")
    f.write(t)
    f.close()

def run_qe(exe, file_prefix, parameters, dry=False, check_done=True):

    if dry: return

    # Dont rerun if already done
    if os.path.isfile(file_prefix+".out"):
        with open(file_prefix+".out") as f:
            s = f.read()
            if "JOB DONE" in s:
                fs = "{0}.out already complete, skipping.\n"
                parameters["out_file"].write(fs.format(file_prefix))
                return

    # Run quantum espresso with specified parallelism
    mpirun = parameters["mpirun"]
    ppn    = parameters["cores_per_node"]
    nodes  = parameters["nodes"]
    np     = nodes * ppn

    # Setup mpi invokation
    if "mpirun" in mpirun:
        mpirun = "{0} -np {1}".format(mpirun, np)
    elif "aprun" in mpirun:
        mpirun = "{0} -n {1}".format(mpirun, np)
    else:
        raise ValueError("Unkown mpirun = {0}".format(mpirun))

    # Check if the ESPRESSO_BIN environment variable is set
    if "ESPRESSO_BIN" in os.environ:
        eb = os.environ["ESPRESSO_BIN"] 
        parameters["out_file"].write("Using QE from {0}\n".format(eb))
        exe = eb + "/" + exe
    
    # Invoke the program 
    qe_flags = "-nk {0}".format(np)
    cmd = "{0} {1} {2} <{3}.in> {3}.out".format(mpirun, exe, qe_flags, file_prefix) 
    parameters["out_file"].write("Running: "+cmd+"\n")
    os.system(cmd)

    # Check calculation completed properly
    if check_done:
        with open(file_prefix+".out") as f:
            if not "JOB DONE" in f.read():
                err = "JOB DONE not found in {0}.out".format(file_prefix)
                parameters["out_file"].write("QE job {0}.in did not complete!\n".format(file_prefix))
                raise Exception("JOB DONE not found in {0}.out".format(file_prefix))

def reduce_to_primitive(parameters):

    try:
        import seekpath
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
        parameters["lattice"]          = prim_geom["primitive_lattice"]
        parameters["atoms"]            = new_atoms
        parameters["bz_path"]          = prim_geom["path"]
        parameters["high_symm_points"] = prim_geom["point_coords"]

        # Convert k-points to cartesian coords in units of
        # 2pi/a0, because that's what q.e uses for some reason
        #recip_prim_lattice = np.array(prim_geom["reciprocal_primitive_lattice"])
        #a0 = np.linalg.norm(parameters["lattice"])
        #for p in parameters["high_symm_points"]:
        #    frac_coords = parameters["high_symm_points"][p]
        #    cart_coords = np.matmul(recip_prim_lattice.T, frac_coords)
        #    parameters["high_symm_points"][p] = cart_coords*a0/(2*np.pi)

        outf = parameters["out_file"]
        outf.write("Successfully reduced to primitive geometry using seekpath.\n")

    except ImportError:
        err  = "Could not import seekpath =>\n"
        err += "  1. We cannot reduce to the primitive geometry\n"
        err += "  2. We cannot obtain Brilloin zone paths\n"
        err += "  3. 2 => we cannot calculate bandstructures\n"
        parameters["out_file"].write(err)
        
        if parameters["require_prim_geom"]:
            ex_mess  = "Could not reduce to primitive geometry/find BZ path. "
            ex_mess += "Perhaps the version of python you're using does not "
            ex_mess += "have access to the seeKpath module."
            raise Exception(ex_mess)

    # Work out qpoint_grid
    if "qpoint_grid" in parameters:
        
        # Warn user we're using an explicit q-point grid
        if "qpoint_spacing" in parameters:
            parameters["out_file"].write(
                "Explicit q-point grid specified, ignoring q-point spacing\n")

    elif "qpoint_spacing" in parameters:
        
        # Work out qpoint_grid from qpoint_spacing
        parameters["qpoint_grid"] = get_kpoint_grid(
            parameters["lattice"], parameters["qpoint_spacing"])

    # Ensure we have at least 1 q-point in each direction
    parameters["qpoint_grid"] = [max(1,q) for q in parameters["qpoint_grid"]]

    if parameters["kpoint_grid"] is None:
        # Work out k-point grid from q-point grid and
        # k-points per qpoint
        kpq = parameters["kpts_per_qpt"]
        qpg = parameters["qpoint_grid"]
        parameters["kpoint_grid"] = [int(q*k) for q,k in zip(qpg, kpq)]
        mess = "Generating k-point grid from q-point grid: {0}x{1}x{2}\n"
    else:
        # Use explicitly specified k-point grid
        mess = "Using explicit k-point grid: {0}x{1}x{2}\n"

    # Tell user the k-point grid we're using and how we got it
    parameters["out_file"].write(mess.format(*parameters["kpoint_grid"]))

    # Return resulting new parameter set
    return parameters

def run(parameters, dry=False, aux_kpts=False):

    # Open the output file
    parameters["out_file"] = open("run.out","w",1)
    parameters["out_file"].write("Dryrun   : {0}\n".format(dry))
    parameters["out_file"].write("Aux kpts : {0}\n".format(aux_kpts))
    max_l = str(max([len(p) for p in parameters]))
    for p in parameters:
        fs = "{0:"+max_l+"."+max_l+"} : {1}\n"
        parameters["out_file"].write(fs.format(p, parameters[p]))

    # Run with the auxilliary kpt grid
    if aux_kpts:
        parameters["out_file"].write("Running auxillary kpoint grid...\n")
        parameters["kpts_per_qpt"] = parameters["aux_kpts"]

    # Reduce to primitive description
    parameters = reduce_to_primitive(parameters)
    
    # Caclulate relaxed geometry
    create_relax_in(parameters)
    run_qe("pw.x", "relax", parameters, dry=dry)
    if not dry:
        relax_data = parse_vc_relax("relax.out")
        parameters["lattice"] = relax_data["lattice"]
        parameters["atoms"]   = relax_data["atoms"]

    # Stop here if we're just doing relaxations
    if parameters["relax_only"]: return

    # Run SCF with the new geometry
    create_scf_in(parameters)
    run_qe("pw.x", "scf", parameters, dry=dry)

    if parameters["irrep_group_size"] > 0:

        # Run elec-phonon prep calculation
        create_elph_in("elph_prep", parameters, irr_range=[0,0])
        run_qe("ph.x", "elph_prep", parameters, dry=dry, check_done=False)

        # Count q-points
        qpoint_count = 0
        with open("elph_prep.out") as elph_prep_f:
            for line in elph_prep_f:
                if "q-points):" in line:
                    qpoint_count = int(line.split("q-points")[0].replace("(",""))
                    break

        parameters["out_file"].write("q-points to calculate: {0}\n".format(qpoint_count))

        # Count irreps
        irrep_counts = {}  
        for i in range(1, qpoint_count+1):
            with open("_ph0/pwscf.phsave/patterns.{0}.xml".format(i)) as pat_file:
                next_line = False
                for l in pat_file:
                    if next_line:
                        irrep_counts[i] = int(l)
                        break
                    if "NUMBER_IRR_REP" in l:
                        next_line = True

        for i in irrep_counts:
            fs = "    irreducible representations for q-point {0}: {1}\n"
            parameters["out_file"].write(fs.format(i, irrep_counts[i]))

        gs = int(parameters["irrep_group_size"])

        # Run elec-phonon calculations for each irrep group
        for q_point in range(1, qpoint_count+1):
            for irr in range(1, irrep_counts[q_point]+1, gs):
                name = "elph_{0}_{1}_{2}".format(q_point, irr, irr+gs-1)
                create_elph_in(name, parameters,
                    irr_range=[irr, min(irr+gs-1, irrep_counts[q_point])], 
                    q_range=[q_point, q_point])
                run_qe("ph.x", name, parameters, dry=dry)

        # Collect phonon results/diagonalise dynamical matrix
        create_elph_in("elph_collect", parameters, force_recover=True)
        run_qe("ph.x", "elph_collect", parameters, dry=dry)

    else:
        
        # Just calculate all electron-phonon stuff in a single step
        create_elph_in("elph_all", parameters)
        run_qe("ph.x", "elph_all", parameters, dry=dry)

    # Delete phonon files after successful elph run
    if parameters["disk_usage"] == "minimal":
        os.system("rm -r _ph0")

    # Convert dynamcial matricies etc to real space
    create_q2r_in(parameters)
    run_qe("q2r.x", "q2r", parameters, dry=dry)

    # Caclulate the phonon density of states
    create_ph_dos_in(parameters)
    run_qe("matdyn.x", "ph_dos", parameters, dry=dry)

    # Extract the eigenvalues
    create_extract_evals_in(parameters)
    run_qe("bands.x", "extract_evals", parameters, dry=dry)

    # Only do the following if we have a brillouin zone path
    if "bz_path" in parameters:

        # Caclulate the phonon bandstructure
        create_ph_bands_in(parameters)
        run_qe("matdyn.x", "ph_bands", parameters, dry=dry)

        # Caculate the electronic bandstructure
        create_bands_in(parameters)
        run_qe("pw.x", "bands", parameters, dry=dry)

        # Re-order bands and calc band-related things
        create_bands_x_in(parameters)
        run_qe("bands.x", "bands.x", parameters, dry=dry)

def is_run_complete():

    # Checks to see if a run is complete by
    # checking the last calculation is done
    if not os.path.isfile("extract_evals.out"):
        return False
    with open("extract_evals.out") as f:
        return "JOB DONE" in f.read()

def run_dir(directory, infile, dry, aux_kpts):
    
    # Run the calculation in the given directory,
    # with the given input file
    os.chdir(directory)
    params = read_parameters(infile)
    run(params, dry=dry, aux_kpts=aux_kpts)

def submit_calc(directory, infile, submit, dry, aux_kpts):
    
    # Submit the calculation in the given directory
    os.chdir(directory)
    params = read_parameters(infile)

    if submit is None:

        # Actually run the calculation
        run(params, dry=dry, aux_kpts=aux_kpts)

    elif submit.startswith("csd3"):
        
        # Choose service level
        sub_file = "slurm_submit_csd3_sl4"
        if "sl3" in submit:
            sub_file = "slurm_submit_csd3_sl3"

        # Submit the calculation to CSD3 via SLURM
        script_dir = os.path.realpath(__file__)
        script_dir = "/".join(script_dir.split("/")[0:-1])

        # Copy the slurm submission script
        sub_script = script_dir+"/"+sub_file
        with open(sub_script) as f:
            sub_text = f.read()

        # Create the submission script with the given core/node count
        with open(directory+"/"+sub_file, "w") as f:
            cores_total = params["cores_per_node"]*params["nodes"]
            f.write(sub_text.format(
                nodes=params["nodes"],
                cores_total=cores_total
                ))

        # Create the python runscript
        r  = "from quantum_espresso_tools.superconductivity.calculate import run_dir\n"
        r += "run_dir('{0}', '{1}', {2}, {3})".format(directory, infile, dry, aux_kpts)
        with open(directory+"/run.py", "w") as f:
            f.write(r)

        # If is a dry run, simply run it, otherwise submit it
        if dry: os.system("python2.7 run.py")
        else:
            if is_run_complete():
                print("{0} already complete, refusing to submit.".format(directory))
            else:
                print("Submitting {0}".format(directory))
                os.system("sbatch "+sub_file)
    else:
        print("Unkown submission system: "+submit)

def calculate(infile, dry=False, submit=None):

    # Base directory and auxillary k-point directory
    base_dir    = os.getcwd()
    params      = read_parameters(infile)
    
    if not params["aux_kpts"] is None:
        # Run auxilliary k-point grid
        # (do this first because it's faster)
        aux_dir = base_dir + "/aux_kpts"
        os.system("mkdir " + aux_dir + " 2>/dev/null")
        os.system("cp "    + infile  + " " + aux_dir)
        submit_calc(aux_dir, infile, submit, dry, True)

    # Run normal k-point grid
    primary_dir = base_dir + "/primary_kpts"
    os.system("mkdir " + primary_dir  + " 2>/dev/null")
    os.system("cp "    + infile + " " + primary_dir)
    submit_calc(primary_dir, infile, submit, dry, False)
