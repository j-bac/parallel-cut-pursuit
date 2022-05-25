import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_prox_tv_cpy import cp_prox_tv_cpy

def cp_prox_tv(Y, first_edge, adj_vertices, edge_weights=None, 
               cp_dif_tol=1e-4, cp_it_max=10, pfdr_rho=1., 
               pfdr_cond_min=1e-2, pfdr_dif_rcd=0., pfdr_dif_tol=None, 
               pfdr_it_max=int(1e4), verbose=int(1e3), max_num_threads=0,
               balance_parallel_split=True, compute_List=False,
               compute_Subgrads=False, compute_Obj=False, compute_Time=False,
               compute_Dif=False):
    """
    Comp, rX, [List, Gtv, Obj, Time, Dif] = cp_prox_tv(Y, first_edge,
            adj_vertices, edge_weights=1.0, cp_dif_tol=1e-4, cp_it_max=10,
            pfdr_rho=1.0, pfdr_cond_min=1e-2, pfdr_dif_rcd=0.0,
            pfdr_dif_tol=1e-2*cp_dif_tol, pfdr_it_max=int(1e4),
            verbose=int(1e3), max_num_threads=0, balance_parallel_split=True,
            compute_List=False, compute_Subgrads=False, compute_Obj=False,
            compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm for proximity operator of total variation
    penalization:

    minimize functional over a graph G = (V, E)

        F(x) = 1/2 ||y - x||^2 + ||x||_d1

    where x, y in R^V, and
          ||x||_d1 = sum_{uv in E} w_uv |x_u - x_v|,

    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type is uint32.

    NOTA: by default, components are identified using uint16 identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    Y - observations, (real) array of length V;
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often
        be taken care of without actually copying data by using transpose();
    first_edge, adj_vertices - graph forward-star representation:
        vertices are numeroted (start at 0) in the order given in Y
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting 
            from the vertex (or, if there are none, starting from the next 
            vertex); (uint32) array of length V + 1, the last value is the
            total number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (uint32)
            array of length E
    edge_weights - (real) array of length E or a scalar for homogeneous weights
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-4 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max - maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
    pfdr_rho - relaxation parameter, 0 < rho < 2
        1 is a conservative value; 1.5 often speeds up convergence
    pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
        corresponds roughly the minimum ratio to the maximum descent metric;
        1e-2 is typical; a smaller value might enhance preconditioning
    pfdr_dif_rcd - reconditioning criterion on iterate evolution;
        a reconditioning is performed if relative changes of the iterate drops
        below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
        away from the solution set and give bad subproblem solutions
    pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol
        1e-2*cp_dif_tol is a conservative value
    pfdr_it_max - maximum number of iterations 1e4 iterations provides enough
        precision for most subproblems
    verbose - if nonzero, display information on the progress, every 'verbose'
        PFDR iterations
    max_num_threads - if greater than zero, set the maximum number of threads
        used for parallelization with OpenMP
    balance_parallel_split - if true, the parallel workload of the split step 
        is balanced; WARNING: this might trade off speed against optimality
    compute_List - report the list of vertices constituting each component
    compute_Subgrads - compute the total variation subgradients
    compute_Obj - compute the objective functional along iterations 
    compute_Time - monitor elapsing time along iterations
    compute_Dif - compute relative evolution along iterations 

    OUTPUTS: List, Gtv, Obj, Time and Dif are optional, set parameters
        compute_List, compute_Subgrads, compute_Obj, compute_Time or
        compute_Dif to True to request them and capture them in output
        variables in that order

    Comp - assignement of each vertex to a component, (uint16) array of
        length V
    rX - values of each component of the minimizer, (real) array of length rV;
        the actual minimizer is then reconstructed as X = rX[Comp];
    List - if requested, list of vertices constituting each component; python
        list of length rV, containing (uint32) arrays of indices
    Gtv - subgradients of the total variation penalization at solution; (real)
        array of length E; if e is the edge (u, v), the subgradient of the
        total variation penalization at vertices (u, v) is (-Gd1[e], Gd1[e])
    Obj - if requested, the values of the objective functional along
        iterations, up to the constant 1/2||Y||^2;
        array of length the actual number of iterations performed + 1;
    Time - if requested, the elapsed time along iterations;
        array of length the actual number of iterations performed + 1
    Dif  - if requested, the iterate evolution along iterations;
        array of length the actual number of iterations performed

    Parallel implementation with OpenMP API.

    L. Landrieu and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn
    Piecewise Constant Functions on General Weighted Graphs, SIIMS, 10, 4,
    1724–1766, 2017.
 
    Hugo Raguet 2021
    """

    # Check numpy arrays: Y, first_edge, adj_vertices, edge_weights, and
    # define float numpy array argument with the right float type if necessary
    if type(Y) != np.ndarray:
        raise TypeError("Cut-pursuit prox TV: argument 'Y' must be a numpy "
                        "array.")

    # real type is determined by Y
    real_t = Y.dtype
    if real_t not in ["float32", "float64"]:
        raise TypeError("Cut-pursuit prox TV: currently, the real numeric type"
                        " must be float32 or float64.")

    if type(first_edge) != np.ndarray or first_edge.dtype != "uint32":
        raise TypeError("Cut-pursuit prox TV: argument 'first_edge' must be a "
                        "numpy array of type uint32.")

    if type(adj_vertices) != np.ndarray or adj_vertices.dtype != "uint32":
        raise TypeError("Cut-pursuit prox TV: argument 'adj_vertices' must be "
                        "a numpy array of type uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit prox TV: argument 'edge_weights' must"
                            " be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    # Determine V and check the graph structure
    V = Y.size
    if first_edge.size != V + 1:
        raise ValueError("Cut-pursuit prox TV: argument 'first_edge' should "
                         "contain |V| + 1 = {0} elements, but {1} are given."
                         .format(V + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float (Y, edge_weights) 
    for name, ar_args in zip(["Y", "edge_weights"], [Y, edge_weights]):
        if ar_args.dtype != real_t:
            raise TypeError("argument '{0}' must be of type '{1}'"
                            .format(name, real_t))

    # Check fortran continuity of all multidimensional numpy array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit prox TV: argument 'Y' must be in "
                        "column-major order (F-contigous).")

    # Convert in float64 all float arguments if needed (cp_dif_tol, pfdr_rho, 
    # pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol) 
    if pfdr_dif_tol is None:
        pfdr_dif_tol = 1e-2*cp_dif_tol
    cp_dif_tol = float(cp_dif_tol)
    pfdr_rho = float(pfdr_rho)
    pfdr_cond_min = float(pfdr_cond_min)
    pfdr_dif_rcd = float(pfdr_dif_rcd)
    pfdr_dif_tol = float(pfdr_dif_tol)
     
    # Convert all int arguments (cp_it_max, pfdr_it_max, verbose) in ints: 
    cp_it_max = int(cp_it_max)
    pfdr_it_max = int(pfdr_it_max)
    verbose = int(verbose)
    max_num_threads = int(max_num_threads)

    # Check type of all booleen arguments (balance_parallel_split,
    # compute_List, compute_Subgrads, compute_Obj, compute_Time, compute_Dif)
    for name, b_args in zip(
            ["balance_parallel_split", "compute_List", "compute_Subgrads",
             "compute_Obj", "compute_Time", "compute_Dif"],
            [ balance_parallel_split ,  compute_List ,  compute_Subgrads ,
              compute_Obj ,  compute_Time ,  compute_Dif ]):
        if type(b_args) != bool:
            raise TypeError("Cut-pursuit prox TV: argument '{0}' must be "
                            "boolean".format(name))
    
    # Call wrapper python in C  
    return cp_prox_tv_cpy(Y, first_edge, adj_vertices, edge_weights,
            cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd,
            pfdr_dif_tol, pfdr_it_max, verbose, max_num_threads,
            balance_parallel_split, real_t == "float64", compute_List,
            compute_Subgrads, compute_Obj, compute_Time, compute_Dif) 
