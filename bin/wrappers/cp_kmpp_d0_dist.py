import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_kmpp_d0_dist_cpy import cp_kmpp_d0_dist_cpy

def cp_kmpp_d0_dist(loss, Y, first_edge, adj_vertices, edge_weights=None, 
                    vert_weights=None, coor_weights=None, cp_dif_tol=1e-3,
                    cp_it_max=10, K=2, split_iter_num=2, split_damp_ratio=1.0,
                    kmpp_init_num=3, kmpp_iter_num=3, min_comp_weight = 0.0,
                    verbose=True, max_num_threads=0,
                    balance_parallel_split=True, compute_List=False,
                    compute_Obj=False, compute_Time=False, compute_Dif=False):

    """
    Comp, rX, [List, Obj, Time, Dif] = cp_kmpp_d0_dist(loss, Y, first_edge,
            adj_vertices, edge_weights=None, vert_weights=None,
            coor_weights=None, cp_dif_tol=1e-3, cp_it_max=10, K=2,
            split_iter_num=2, split_damp_ratio=1.0, kmpp_init_num=3,
            kmpp_iter_num=3, min_comp_weight=0.0, verbose=True,
            max_num_threads=0, balance_parallel_split=True, compute_List=False,
            compute_Obj=False, compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm with d0 (weighted contour length) penalization, with
    a loss akin to a distance:

    minimize functional over a graph G = (V, E)

        F(x) = sum_v loss(y_v, x_v) + ||x||_d0

    where for each vertex, y_v and x_v are D-dimensional vectors, the loss is
    either the sum of square differences or smoothed Kullback-Leibler 
    divergence (equivalent to cross-entropy in this formulation); see the 
    'loss' attribute, and ||x||_d0 = sum_{uv in E} w_d0_uv ,

    using greedy cut-pursuit approach with splitting initialized with 
    k-means++.

    Available data-fidelity loss include:

    quadratic:
        f(x) = ||y - x||_{l2,W}^2 ,
    where W is a diagonal metric (separable product along ℝ^V and ℝ^D),
    that is ||y - x||_{l2,W}^2 = sum_{v in V} w_v ||x_v - y_v||_{l2,M}^2
                               = sum_{v in V} w_v sum_d m_d (x_vd - y_vd)^2;

    (smoothed, weighted) Kullback-Leibler divergence (equivalent to
    cross-entropy) on the probability simplex:
        f(x) = sum_v w_v KLs_m(x_v, y_v),
    with KLs(y_v, x_v) = KL_m(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
        KL is the regular Kullback-Leibler divergence,
        u is the uniform discrete distribution over {1,...,D}, and
        s = loss is the smoothing parameter
        m is a diagonal metric weighting the coordinates;
    it yields
        KLs_m(y_v, x_v) = - H_m(s u + (1 - s) y_v)
            - sum_d m_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
    where H_m is the (weighted) entropy, that is H_m(s u + (1 - s) y_v)
        = - sum_d m_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
    note that the choosen order of the arguments in the Kullback--Leibler
    does not favor the entropy of x (H_m(s u + (1 - s) y_v) is a constant),
    hence this loss is actually equivalent to cross-entropy.
 
    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type is uint32.

    NOTA: by default, components are identified using uint16 identifiers;
    this can be easily changed in the wrapper source if more than 65535 
    components are expected (recompilation is necessary)

    loss - 1 for quadratic, 0 < loss < 1 for smoothed Kullback-Leibler
    Y - observations, (real) D-by-V array;
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often
        be taken care of without actually copying data by using transpose();
        for Kullback-Leibler loss, the value at each vertex must lie on the
        probability simplex
    first_edge, adj_vertices - forward-star graph representation:
        vertices are numeroted (start at 0) in the order they are given in Y;
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting from 
            the vertex (or, if there are none, starting from the next vertex);
            (uint32) array of length V+1, the last value is the total number of
            edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (uint32)
          array of length E
    edge_weights - (real) array of length E or scalar for homogeneous weights
    vert_weights - weights on vertices (w_v in above notations)
        (real) array of length V or empty for no weights
    coor_weights - weights on coordinates (m_d above notations)
        (real) array of length D or empty for no weights
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (that is, relative dissimilarity measures defined by 
        the choosen loss between succesive iterates and between current iterate
        and observation) is less than dif_tol; 1e-3 is a typical value
    cp_it_max  - maximum number of iterations (graph cut, subproblem, merge)
        10 cuts solve accurately most problems
    K - number of alternative values considered in the split step
    split_iter_num - number of partition-and-update iterations in the split 
        step
    split_damp_ratio - edge weights damping for favoring splitting; edge
        weights increase in linear progression along partition-and-update
        iterations, from this ratio up to original value; real scalar between 0
        and 1, the latter meaning no damping
    kmpp_init_num - number of random k-means initializations in the split step
    kmpp_iter_num - number of k-means iterations in the split step
    min_comp_weight - minimum total weight (number of vertices if no weights
        are given on the vertices) that a component is allowed to have;
        components with smaller weights are merged with adjacent components
    verbose - if true, display information on the progress
    max_num_threads - if greater than zero, set the maximum number of threads
        used for parallelization with OpenMP
    balance_parallel_split - if true, the parallel workload of the split step 
        is balanced; WARNING: this might trade off speed against optimality
    compute_List - report the list of vertices constituting each component
    compute_Obj  - compute the objective functional along iterations 
    compute_Time - monitor elapsing time along iterations
    compute_Dif  - compute relative evolution along iterations 

    OUTPUTS: List, Obj, Time and Dif are optional, set parameters compute_List,
        compute_Obj, compute_Time, or compute_Difto True to request them and
        capture them in output variables in that order

    Comp - assignement of each vertex to a component, (uint16) array of
        length V 
    rX  - values of each component of the minimizer, (real) array of length rV;
        the actual minimizer is then reconstructed as X = rX[Comp];
    List - if requested, list of vertices constituting each component; python
        list of length rV, containing (uint32) arrays of indices
    Obj - if requested, values of the objective functional along iterations;
        array of length actual number of cut-pursuit iterations performed + 1
    Time - if requested, the elapsed time along iterations; array of length
        actual number of cut-pursuit iterations performed + 1
    Dif - if requested, if requested, the iterate evolution along iterations;
        array of length actual number of cut-pursuit iterations performed
 
    Parallel implementation with OpenMP API.

    L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
    piecewise constant functions on general weighted graphs, SIAM Journal on
    Imaging Science, 10(4):1724-1766, 2017

    L. Landrieu et al., A structured regularization framework for spatially
    smoothing semantic labelings of 3D point clouds, ISPRS Journal of
    Photogrammetry and Remote Sensing, 132:102-118, 2017

    Baudoin Camille 2019, Raguet Hugo 2021
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if type(Y) != np.ndarray:
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be a "
                        "numpy array.")

    if Y.size > 0 and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.size > 0 and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be a "
                        "nonempty numpy array of type float32 or float64.") 
    
    # Convert in numpy array scalar entry: Y, first_edge, adj_vertices, 
    # edge_weights, vert_weights, coor_weights and define float numpy array
    # argument with the right float type, if empty:
    if type(first_edge) != np.ndarray or first_edge.dtype != "uint32":
        raise TypeError("Cut-pursuit d0 distance: argument 'first_edge' must "
                        "be a numpy array of type uint32.")

    if type(adj_vertices) != np.ndarray or adj_vertices.dtype != "uint32":
        raise TypeError("Cut-pursuit d0 distance: argument 'adj_vertices' "
                        "must be a numpy array of type uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'edge_weights' "
                            "must be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(vert_weights) != np.ndarray:
        if type(vert_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'vert_weights' "
                            "must be a scalar or a numpy array.")
        elif vert_weights != None:
            vert_weights = np.array([vert_weights], dtype=real_t)
        else:
            vert_weights = np.array([], dtype=real_t)

    if type(coor_weights) != np.ndarray:
        if type(coor_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'coor_weights' "
                            "must be a scalar or a numpy array.")
        elif coor_weights != None:
            coor_weights = np.array([coor_weights], dtype=real_t)
        else:
            coor_weights = np.array([], dtype=real_t)

    # Determine V and check graph structure 
    if Y.ndim > 1:
        V = Y.shape[1]
    else:
        V = Y.shape[0]
        
    if first_edge.size != V + 1 :
        raise ValueError("Cut-pursuit d0 distance: argument 'first_edge'"
                         "should contain |V| + 1 = {0} elements, "
                         "but {1} are given".format(V + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float (Y, edge_weights,
    # vert_weights, coor_weights) 
    for name, ar_args in zip(
            ["Y", "edge_weights", "vert_weights", "coor_weights"],
            [ Y ,  edge_weights ,  vert_weights ,  coor_weights ]):
        if ar_args.dtype != real_t:
            raise TypeError("Cut-pursuit d0 distance: argument '{0}' must be "
                            "of type '{1}'".format(name, real_t))

    # Check fortran continuity of all multidimensional numpy array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be in "
                        "column-major order (F-contigous).")

    # Convert in float64 all float arguments if needed (loss, cp_dif_tol) 
    loss = float(loss)
    cp_dif_tol = float(cp_dif_tol)
    split_damp_ratio = float(split_damp_ratio)
     
    # Convert all int arguments (cp_it_max, K, split_iter_num, kmpp_init_num, 
    # kmpp_iter_num, verbose) in ints: 
    cp_it_max = int(cp_it_max)
    K = int(K)
    split_iter_num = int(split_iter_num)
    kmpp_init_num = int(kmpp_init_num)
    kmpp_iter_num = int(kmpp_iter_num)
    max_num_threads = int(max_num_threads)

    # Check type of all booleen arguments
    for name, b_args in zip(
        ["verbose", "balance_parallel_split", "compute_Obj", "compute_Time", 
         "compute_Dif"],
        [ verbose ,  balance_parallel_split ,  compute_Obj ,  compute_Time , 
          compute_Dif ]):
        if type(b_args) != bool:
            raise TypeError("Cut-pursuit d0 distance: argument '{0}' must be "
                            "boolean".format(name))

    # Call wrapper python in C  
    return cp_kmpp_d0_dist_cpy(loss, Y, first_edge, adj_vertices, edge_weights,
            vert_weights, coor_weights, cp_dif_tol, cp_it_max, K,
            split_iter_num, split_damp_ratio, kmpp_init_num, kmpp_iter_num,
            min_comp_weight, verbose, max_num_threads, balance_parallel_split,
            real_t == "float64", compute_List, compute_Obj, compute_Time,
            compute_Dif)
