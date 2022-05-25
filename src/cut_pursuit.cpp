/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <algorithm>
#include "cut_pursuit.hpp"

#define ZERO ((real_t) 0.0) // avoid conversions
#define ONE ((size_t) 1) // avoid overflows
#define TWO ((size_t) 2) // avoid overflows
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)

/** specific flags **/
#define MAX_NUM_COMP (std::numeric_limits<comp_t>::max())
/* use maximum number of components; no component can have this identifier */
#define NOT_ASSIGNED MAX_NUM_COMP
#define ASSIGNED ((comp_t) 0)
#define ASSIGNED_ROOT ((comp_t) 1) // must differ from ASSIGNED
#define ASSIGNED_ROOT_SAT ((comp_t) 2) // must differ from ASSIGNED_ROOT
#define NOT_SATURATED ((comp_t) 1) // must differ from ASSIGNED
/* use maximum number of edges; no edge can have this identifier */
#define NO_EDGE (std::numeric_limits<index_t>::max())
#define NOT_ISOLATED NO_EDGE
#define ISOLATED ((index_t) 0)

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP Cp<real_t, index_t, comp_t, value_t>

using namespace std; 

TPL CP::Cp(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D)
    : V(V), E(E), first_edge(first_edge), adj_vertices(adj_vertices), D(D)
{
    /* real type with infinity is handy */
    static_assert(numeric_limits<real_t>::has_infinity,
        "Cut-pursuit: real_t must be able to represent infinity.");

    /* edge activation */
    edge_status = (Edge_status*) malloc_check(sizeof(Edge_status)*E);
    for (index_t e = 0; e < E; e++){ bind(e); }

    /* reduced graph **/
    rV = 1; rE = 0;
    last_rV = 0;
    saturated_comp = 0;
    saturated_vert = 0;
    edge_weights = nullptr;
    homo_edge_weight = 1.0;
    comp_assign = last_comp_assign = nullptr;
    comp_list = first_vertex = index_in_comp = nullptr;
    is_saturated = nullptr;
    reduced_edge_weights = nullptr;
    reduced_edges = nullptr;
    elapsed_time = nullptr;
    objective_values = iterate_evolution = nullptr;
    rX = last_rX = nullptr;

    /* some algorithmic parameters */
    it_max = 10; verbose = 1000;
    dif_tol = ZERO;
    eps = numeric_limits<real_t>::epsilon();
    monitor_evolution = false;

    max_num_threads = omp_get_max_threads();
    balance_par_split = max_num_threads > 1 &&
        compute_num_threads(maxflow_complexity()) > 1;
}

TPL CP::~Cp()
{
    free(edge_status);
    free(comp_assign); free(last_comp_assign);
    free(first_vertex);
    free(comp_list);
    free(index_in_comp);
    free(is_saturated);
    free(reduced_edges); free(reduced_edge_weights);
    free(rX); free(last_rX);
}

TPL void CP::reset_edges()
{ for (index_t e = 0; e < E; e++){ bind(e); } }

TPL void CP::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight)
{
    this->edge_weights = edge_weights;
    this->homo_edge_weight = homo_edge_weight;
}

TPL void CP::set_monitoring_arrays(real_t* objective_values,
    double* elapsed_time, real_t* iterate_evolution)
{
    this->objective_values = objective_values;
    this->elapsed_time = elapsed_time;
    this->iterate_evolution = iterate_evolution;
    if (iterate_evolution){ monitor_evolution = true; }
}

TPL void CP::set_components(comp_t rV, comp_t* comp_assign)
{
    if (rV > 1 && !comp_assign){
        cerr << "Cut-pursuit: if an initial number of components greater than "
            "one is given, components assignment must be provided." << endl;
        exit(EXIT_FAILURE);
    }
    this->rV = rV;
    this->comp_assign = comp_assign;
}

TPL void CP::set_cp_param(real_t dif_tol, int it_max, int verbose, real_t eps)
{
    this->dif_tol = dif_tol;
    if (dif_tol > ZERO){ monitor_evolution = true; }
    this->it_max = it_max;
    this->verbose = verbose;
    this->eps = ZERO < dif_tol && dif_tol < eps ? dif_tol : eps;
}

TPL void CP::set_parallel_param(int max_num_threads, bool balance_par_split)
{
    if (max_num_threads <= 0){ max_num_threads = omp_get_max_threads(); }
    this->max_num_threads = max_num_threads;
    this->balance_par_split = balance_par_split && max_num_threads > 1
        && compute_num_threads(split_complexity()) > 1;
}

TPL comp_t CP::get_components(comp_t** comp_assign, index_t** first_vertex,
    index_t** comp_list)
{
    if (comp_assign){ *comp_assign = this->comp_assign; }
    if (first_vertex){ *first_vertex = this->first_vertex; }
    if (comp_list){ *comp_list = this->comp_list; }
    return this->rV;
}

TPL index_t CP::get_reduced_graph(comp_t** reduced_edges,
    real_t** reduced_edge_weights)
{
    
    if (reduced_edges){
        if (!this->reduced_edges){ compute_reduced_graph(); }
        *reduced_edges = this->reduced_edges;
    }
    if (reduced_edge_weights){
        *reduced_edge_weights = this->reduced_edge_weights;
    }
    return this->rE;
}

TPL value_t* CP::get_reduced_values(){ return rX; }

TPL void CP::set_reduced_values(value_t* rX){ this->rX = rX; }

TPL int CP::cut_pursuit(bool init)
{
    int it = 0;
    double timer = 0.0;
    real_t dif = real_inf();

    chrono::steady_clock::time_point start;
    if (elapsed_time){ start = chrono::steady_clock::now(); }
    if (init){
        if (verbose){ cout << "Cut-pursuit initialization:" << endl; }
        initialize();
        if (objective_values){ objective_values[0] = compute_objective(); }
    }

    while (true){
        if (elapsed_time){ elapsed_time[it] = timer = monitor_time(start); }
        if (verbose){ print_progress(it, dif, timer); }
        if (it == it_max || dif <= dif_tol){ break; }

        if (verbose){
            cout << "Cut-pursuit iteration " << it + 1 << " (max. " << it_max
                << "): " << endl;
        }

        if (verbose){ cout << "\tSplit... " << flush; }
        index_t activation = split();
        if (verbose){
            cout << activation << " new activated edge(s)." << endl;
        }

        if (!activation){ /* do not recompute reduced problem */
            saturated_comp = rV;
            saturated_vert = V;
            if (dif_tol > ZERO || iterate_evolution){
                dif = ZERO;
                if (iterate_evolution){ iterate_evolution[it] = dif; }
            }

            it++;

            if (objective_values){
                objective_values[it] = objective_values[it - 1];
            }
            continue;
        }

        /* store previous component assignment */
        last_comp_assign = (comp_t*) malloc_check(sizeof(comp_t)*V);
        for (index_t v = 0; v < V; v++){
            last_comp_assign[v] = comp_assign[v];
        }
        last_rV = rV;
        if (monitor_evolution){ /* store also last iterate values */
            last_rX = (value_t*) malloc_check(sizeof(value_t)*D*rV);
            for (size_t i = 0; i < D*rV; i++){ last_rX[i] = rX[i]; }
        }
        /* reduced graph and components will be updated */
        free(rX); rX = nullptr;

        if (verbose){ cout << "\tCompute connected components... " << flush; }
        compute_connected_components();
        if (verbose){
            cout << rV << " connected component(s), " << saturated_comp <<
                " saturated." << endl;
        }

        if (verbose){ cout << "\tCompute reduced graph... " << flush; }
        compute_reduced_graph();
        if (verbose){ cout << rE << " reduced edge(s)." << endl; }

        if (verbose){ cout << "\tSolve reduced problem: " << endl; }
        rX = (value_t*) malloc_check(sizeof(value_t)*D*rV);
        solve_reduced_problem();

        /* compute evolution before the merge step because parallel balancing
         * components will be merged, resulting in underestimated evolution */
        if (monitor_evolution){
            dif = compute_evolution(dif_tol > ZERO || iterate_evolution);
            if (iterate_evolution){ iterate_evolution[it] = dif; }
            free(last_rX); last_rX = nullptr;
        }

        if (verbose){ cout << "\tMerge... " << flush; }
        index_t deactivation = merge();
        if (verbose){
            cout << deactivation << " deactivated edge(s)." << endl;
        }

        free(last_comp_assign); last_comp_assign = nullptr;

        it++;

        if (objective_values){ objective_values[it] = compute_objective(); }

        free(reduced_edges); reduced_edges = nullptr;
        free(reduced_edge_weights); reduced_edge_weights = nullptr;

    } /* endwhile true */

    return it;
}

TPL double CP::monitor_time(chrono::steady_clock::time_point start)
{ 
    using namespace chrono;
    steady_clock::time_point current = steady_clock::now();
    return ((current - start).count()) * steady_clock::period::num
               / static_cast<double>(steady_clock::period::den);
}

TPL void CP::print_progress(int it, real_t dif, double timer)
{
    if (it && (dif_tol > ZERO || iterate_evolution)){
        cout.precision(2);
        cout << scientific << "\trelative iterate evolution " << dif
            << " (tol. " << dif_tol << ")\n";
    }
    cout << "\t" << rV << " connected component(s), " << saturated_comp <<
        " saturated, and at most " << rE << " reduced edge(s).\n";
    if (timer > 0.0){
        cout.precision(1);
        cout << fixed << "\telapsed time " << timer << " s.\n";
    }
    cout << endl;
}

TPL void CP::single_connected_component()
{
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*2);
    first_vertex[0] = 0; first_vertex[1] = V;
    rV = 1;
    for (index_t v = 0; v < V; v++){ comp_assign[v] = 0; }
    for (index_t v = 0; v < V; v++){ comp_list[v] = v; }
}

TPL void CP::assign_connected_components()
{
    /* activate edges between components */
    #pragma omp parallel for schedule(static) NUM_THREADS(E, V)
    for (index_t v = 0; v < V; v++){
        comp_t rv = comp_assign[v];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (rv != comp_assign[adj_vertices[e]]){ cut(e); }
        }
    }

    /* translate 'comp_assign' into dual representation 'comp_list' */
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*(rV + ONE));
    for (comp_t rv = 0; rv < rV + ONE; rv++){ first_vertex[rv] = 0; }
    for (index_t v = 0; v < V; v++){ first_vertex[comp_assign[v] + 1]++; }
    for (comp_t rv = 1; rv < rV - 1; rv++){
        first_vertex[rv + 1] += first_vertex[rv];
    }
    for (index_t v = 0; v < V; v++){
        comp_list[first_vertex[comp_assign[v]]++] = v;
    }
    for (comp_t rv = rV; rv > 0; rv--){
        first_vertex[rv] = first_vertex[rv - 1];
    }
    first_vertex[0] = 0;
}

TPL void CP::get_bind_reverse_edges(comp_t rv, index_t*& first_edge_r,
        index_t*& adj_vertices_r)
{
    const index_t* comp_list_rv = comp_list + first_vertex[rv];
    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
    first_edge_r = (index_t*) malloc_check(sizeof(index_t)*(comp_size + ONE));
    /* set index of each vertex in the component */
    for (index_t i = 0; i < comp_size; i++){
        index_in_comp[comp_list_rv[i]] = i;
    }
    /* count reverse edges for each vertex (shift by one index) */
    for (index_t i = 0; i < comp_size + ONE; i++){ first_edge_r[i] = 0; }
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_bind(e)){ /* keep only binding edges */
                first_edge_r[index_in_comp[adj_vertices[e]] + 1]++;
            }
        }
    }
    /* cumulative sum for actual first binding edge id for each vertex */
    first_edge_r[0] = 0;
    for (index_t i = 2; i < comp_size + ONE; i++){
        first_edge_r[i] += first_edge_r[i - 1];
    }
    /* store adjacent vertices, using previous sum as starting indices */
    adj_vertices_r = (index_t*)
        malloc_check(sizeof(index_t)*first_edge_r[comp_size]);
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_bind(e)){
                index_t j = index_in_comp[adj_vertices[e]];
                index_t e_r = first_edge_r[j]++;
                adj_vertices_r[e_r] = v;
            }
        }
    }
    /* first reverse edges have been shifted in the process, shift back */
    for (index_t i = comp_size; i > 0; i--){
        first_edge_r[i] = first_edge_r[i - 1];
    }
    first_edge_r[0] = 0;
}

TPL void CP::compute_connected_components()
{
    /**  new connected components hierarchically derives from previous ones,
     **  we can thus compute them in parallel along previous components  **/

    /* auxiliary variables for parallel region */
    comp_t saturated_comp_par = 0;
    index_t saturated_vert_par = 0;
    index_t tmp_rV = 0; // identify and count components, prevent overflow

    /** there is need to scan all edges involving a given vertex without
     * running through all edges of the graph, so we create the list of
     * 'reverse edges' within each component; to facilitate this, we keep the
     * index of each vertex within its component **/
    index_in_comp = (index_t*) malloc_check(sizeof(index_t)*V);

    #pragma omp parallel for schedule(dynamic) NUM_THREADS(2*E, rV) \
        reduction(+:tmp_rV, saturated_comp_par, saturated_vert_par)
    for (comp_t rv = 0; rv < rV; rv++){
        index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];

        if (is_saturated[rv]){ /* component stays the same */
            index_t i = first_vertex[rv];
            comp_assign[comp_list[i]] = ASSIGNED_ROOT_SAT; // flag the root
            for (i++; i < first_vertex[rv + 1]; i++){
                comp_assign[comp_list[i]] = ASSIGNED;
            }
            saturated_comp_par++;
            saturated_vert_par += comp_size;
            tmp_rV++;
            continue;
        } /* else component has been split */

        /* cleanup assigned components */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            comp_assign[comp_list[i]] = NOT_ASSIGNED;
        }

        /* get reverse binding edges for breadth-first search */
        index_t *first_edge_r, *adj_vertices_r;
        get_bind_reverse_edges(rv, first_edge_r, adj_vertices_r);

        /* auxiliary component list for reordering vertices */
        index_t* tmp_comp_list_rv = (index_t*)
            malloc_check(sizeof(index_t)*comp_size);

        /**  compute the connected components  **/
        index_t i = 0, j = 0;
        for (index_t k = first_vertex[rv]; k < first_vertex[rv + 1]; k++){
            index_t u = comp_list[k];
            if (comp_assign[u] != NOT_ASSIGNED){ continue; }
            comp_assign[u] = ASSIGNED_ROOT; // flag a component's root
            /* put in connected components list */
            tmp_comp_list_rv[j++] = u;
            while (i < j){ /* breadth-first search */
                index_t v = tmp_comp_list_rv[i++];
                /* add neighbors to the connected component list */
                index_t e = first_edge[v];
                index_t l = index_in_comp[v];
                const index_t* adj_vert = adj_vertices;
                while (adj_vert == adj_vertices || e < first_edge_r[l + 1]){
                    if (adj_vert == adj_vertices){
                        if (e == first_edge[v + 1]){
                            e = first_edge_r[l];
                            adj_vert = adj_vertices_r;
                            continue;
                        }else if (!is_bind(e)){
                            e++; continue; 
                        }
                    }
                    index_t w = adj_vert[e];
                    if (comp_assign[w] == NOT_ASSIGNED){
                        comp_assign[w] = ASSIGNED;
                        tmp_comp_list_rv[j++] = w;
                    }
                    e++;
                }
            } /* the current connected component is complete */
            tmp_rV++;
        }
        free(first_edge_r); free(adj_vertices_r);

        index_t* comp_list_rv = comp_list + first_vertex[rv];
        for (index_t i = 0; i < comp_size; i++){
            comp_list_rv[i] = tmp_comp_list_rv[i];
        }

        free(tmp_comp_list_rv);
    }

    free(index_in_comp); index_in_comp = nullptr;

    saturated_comp = saturated_comp_par;
    saturated_vert = saturated_vert_par;

    if (tmp_rV > MAX_NUM_COMP){
        cerr << "Cut-pursuit: number of components (" << tmp_rV << ") greater "
            << "than can be represented by comp_t (" << MAX_NUM_COMP << ")"
            << endl;
        exit(EXIT_FAILURE);
    }

    /**  update components lists, assignments and saturation  **/
    rV = tmp_rV;
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*(rV + ONE));
    free(is_saturated); 
    is_saturated = (bool*) malloc_check(sizeof(index_t)*rV);
    
    comp_t rv = (comp_t) -1;
    for (index_t i = 0; i < V; i++){
        index_t v = comp_list[i];
        if (comp_assign[v] == ASSIGNED_ROOT ||
            comp_assign[v] == ASSIGNED_ROOT_SAT){
            first_vertex[++rv] = i;
            is_saturated[rv] = comp_assign[v] == ASSIGNED_ROOT_SAT;
        }
        comp_assign[v] = rv;
    }
    first_vertex[rV] = V;
}

TPL void CP::compute_reduced_graph()
/* this could actually be parallelized, but is it worth the pain? */
{
    free(reduced_edges);
    free(reduced_edge_weights);

    if (rV == 1){ /* reduced graph only edge from the component to itself
                   * this is only useful for solving reduced problems with
                   * certain implementations where isolated vertices must be
                   * linked to themselves */
        rE = 1;
        reduced_edges = (comp_t*) malloc_check(sizeof(comp_t)*2);
        reduced_edges[0] = reduced_edges[1] = 0;
        reduced_edge_weights = (real_t*) malloc_check(sizeof(real_t)*1);
        reduced_edge_weights[0] = eps;
        return; 
    }

    /* to avoid allocating rV*(rV - 1)/2, we work component by component;
     * when dealing with component ru, reduced_edge_to[rv] is the identifier of
     * the reduced edge ru -> rv, or NO_EDGE if the edge is not created yet */
    index_t* reduced_edge_to = (index_t*) malloc_check(sizeof(index_t)*rV);
    /* this will also be used to indicate isolated vertices */
    index_t* is_isolated = reduced_edge_to;
    for (comp_t rv = 0; rv < rV; rv++){ is_isolated[rv] = ISOLATED; }

    /**  get all active (cut) edges linking a component to another  **/
    index_t* first_active_edge = (index_t*)
        malloc_check(sizeof(index_t)*(rV + ONE));
    /* count the number of such edges for each component (ind shift by one) */
    for (comp_t rv = 0; rv < rV + ONE; rv++){ first_active_edge[rv] = 0; }
    for (index_t v = 0; v < V; v++){
        comp_t ru = comp_assign[v];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (!is_bind(e) && EDGE_WEIGHTS_(e) > ZERO){ 
                comp_t rv = comp_assign[adj_vertices[e]]; 
                if (ru != rv){
                    /* a nonzero edge involving ru and rv exists */
                    is_isolated[ru] = is_isolated[rv] = NOT_ISOLATED;
                    if (ru < rv){ // count only undirected edges
                        first_active_edge[ru + 1]++;
                    }else{
                        first_active_edge[rv + 1]++;
                    }
                }
            }
        }
    }
    /* cumulative sum, giving first active edge id for each vertex */
    for (comp_t rv = 2; rv < rV + ONE; rv++){
        first_active_edge[rv] += first_active_edge[rv - 1];
    }
    /* store adjacent components and edge weights using previous sum as
     * starting indices */
    comp_t* adj_components = (comp_t*)
        malloc_check(sizeof(comp_t)*first_active_edge[rV]);
    real_t* active_edge_weights = edge_weights || balance_par_split ?
        (real_t*) malloc_check(sizeof(real_t)*first_active_edge[rV]) : nullptr;
    for (index_t v = 0; v < V; v++){
        comp_t ru = comp_assign[v];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (!is_bind(e) && EDGE_WEIGHTS_(e) > ZERO){
                comp_t rv = comp_assign[adj_vertices[e]];
                index_t ae = NO_EDGE;
                if (ru < rv){ // count only undirected edges
                    ae = first_active_edge[ru]++;
                    adj_components[ae] = rv; 
                }else if (rv < ru){
                    ae = first_active_edge[rv]++;
                    adj_components[ae] = ru; 
                }
                if (active_edge_weights && ae != NO_EDGE){
                    /* infinite weights to parallel separation ensure merge */
                    active_edge_weights[ae] = is_par_sep(e) ? real_inf() :
                        EDGE_WEIGHTS_(e);
                }
            }
        }
    }
    /* first active edges have been shifted in the process, shift back */
    for (comp_t rv = rV; rv > 0; rv--){
        first_active_edge[rv] = first_active_edge[rv - 1];
    }
    first_active_edge[0] = 0;

    /* temporary buffer size */
    size_t bufsize = rE > rV * (double) E/V ? rE : rV * (double) E/V;

    reduced_edges = (comp_t*) malloc_check(sizeof(comp_t)*2*bufsize);
    reduced_edge_weights = (real_t*) malloc_check(sizeof(real_t)*bufsize);

    rE = 0; // current number of reduced edges
    index_t last_rE = 0; // keep track of number of processed edges
    for (comp_t ru = 0; ru < rV; ru++){ /* iterate over the components */

        if (is_isolated[ru] == ISOLATED){ /* this is only useful for solving
            * reduced problems with certain implementations where isolated
            * vertices must be linked to themselves */
            if (rE == bufsize){ // reach buffer size
                bufsize += bufsize/2 + 1;
                reduced_edges = (comp_t*) realloc_check(reduced_edges,
                    sizeof(comp_t)*2*bufsize);
                reduced_edge_weights = (real_t*) realloc_check(
                    reduced_edge_weights, sizeof(real_t)*bufsize);
            }
            reduced_edges[TWO*rE] = reduced_edges[TWO*rE + 1] = ru;
            reduced_edge_weights[rE++] = eps;
            continue;
        }

        for (index_t ae = first_active_edge[ru];
             ae < first_active_edge[ru + 1]; ae++){
            real_t edge_weight = active_edge_weights ? active_edge_weights[ae]
                                                     : homo_edge_weight;
            comp_t rv = adj_components[ae];
            index_t re = reduced_edge_to[rv];
            if (re == NO_EDGE){ // a new edge must be created
                if (rE == bufsize){ // reach buffer size
                    bufsize += bufsize/2 + 1;
                    reduced_edges = (comp_t*) realloc_check(reduced_edges,
                        sizeof(comp_t)*2*bufsize);
                    reduced_edge_weights = (real_t*) realloc_check(
                        reduced_edge_weights, sizeof(real_t)*bufsize);
                }
                reduced_edges[TWO*rE] = ru;
                reduced_edges[TWO*rE + 1] = rv;
                reduced_edge_weights[rE] = edge_weight;
                reduced_edge_to[rv] = rE++;
            }else{ /* edge already exists */
                reduced_edge_weights[re] += edge_weight;
            }
        }

        /* reset reduced_edge_to */
        for (; last_rE < rE; last_rE++){
            reduced_edge_to[reduced_edges[TWO*last_rE + 1]] = NO_EDGE;
        }

    }

    free(adj_components);
    free(active_edge_weights);
    free(first_active_edge);
    free(reduced_edge_to);

    if (bufsize > rE){
        reduced_edges = (comp_t*) realloc_check(reduced_edges,
            sizeof(comp_t)*2*rE);
        reduced_edge_weights = (real_t*) realloc_check(reduced_edge_weights,
            sizeof(real_t)*rE);
    }
}

TPL void CP::initialize()
{
    free(rX); 
    if (!comp_assign){
        comp_assign = (comp_t*) malloc_check(sizeof(comp_t)*V);
    }
    if (!comp_list){
        comp_list = (index_t*) malloc_check(sizeof(index_t)*V);
    }

    last_rV = 0;

    reset_edges();

    if (rV > 1){ assign_connected_components(); }
    else{ single_connected_component(); }

    /* start with no saturated component */
    free(is_saturated);
    is_saturated = (bool*) malloc_check(sizeof(bool)*rV);
    for (comp_t rv = 0; rv < rV; rv++){ is_saturated[rv] = false; }

    compute_reduced_graph();
    rX = (value_t*) malloc_check(sizeof(value_t)*D*rV);
    solve_reduced_problem();
    // merge(); no point in challenging assignment provided by user
}

TPL int CP::balance_parallel_split(comp_t& rV_new, comp_t& rV_big, 
        index_t*& first_vertex_big)
{
    /**  sort components by decreasing size  **/
    if (max_num_threads > 1){ // sort even if no balancing is required
        /* get component sizes */
        index_t* comp_sizes = (index_t*) malloc_check(sizeof(index_t)*rV);
        for (comp_t rv = 0; rv < rV; rv++){
            /* saturated components need no processing */
            comp_sizes[rv] = is_saturated[rv] ? 0 :
                first_vertex[rv + 1] - first_vertex[rv];
        }
        /* get sorting permutation indices */
        comp_t* sort_comp = (comp_t*) malloc_check(sizeof(comp_t)*rV);
        for (comp_t rv = 0; rv < rV; rv++){ sort_comp[rv] = rv; }
        /* sorting can be parallelized as well...
         * libstdc++ users can simply compile with -D_GLIBCXX_PARALLEL
         * in which case, omp_set_num_threads() will determine the number of
         * threads used; scaling linearly with rV seems to work best */ 
        omp_set_num_threads(compute_num_threads(rV));
        sort(sort_comp, sort_comp + rV,
            [comp_sizes] (comp_t ru, comp_t rv) -> bool
            { return comp_sizes[ru] > comp_sizes[rv]; }); // decreasing order
        omp_set_num_threads(omp_get_num_procs());
        /* reorder saturation */
        for (comp_t rv = 0; rv < rV; rv++){
            is_saturated[rv] = !comp_sizes[sort_comp[rv]];
        }
        /* reorder components list */
        index_t* tmp_comp_list = (index_t*) malloc_check(sizeof(index_t)*V);
        index_t* tmp_first_vertex = comp_sizes; /* reuse storage */
        index_t i = 0;
        for (comp_t rv = 0; rv < rV; rv++){
            comp_t sort_rv = sort_comp[rv];
            tmp_first_vertex[rv] = i;
            for (index_t j = first_vertex[sort_rv];
                 j < first_vertex[sort_rv + 1]; j++){
                tmp_comp_list[i++] = comp_list[j];
            }
        }
        for (index_t v = 0; v < V; v++){ comp_list[v] = tmp_comp_list[v]; }
        for (comp_t rv = 0; rv < rV; rv++){
            first_vertex[rv] = tmp_first_vertex[rv];
        }
        free(tmp_comp_list);
        free(comp_sizes); /* also storage of tmp_first_vertex */
        /* reorder component values */
        value_t* tmp_rX = (value_t*) malloc_check(sizeof(value_t)*D*rV);
        for (comp_t rv = 0; rv < rV; rv++){
            value_t* tmp_rXv = tmp_rX + D*rv;
            value_t* rXv = rX + D*sort_comp[rv];
            for (size_t d = 0; d < D; d++){ tmp_rXv[d] = rXv[d]; }
        }
        free(rX);
        rX = tmp_rX;

        free(sort_comp);
    }

    if (!balance_par_split){
        rV_new = 0; rV_big = 0;
        return compute_num_threads(split_complexity(), rV);
    }

    int num_thrds = compute_num_threads(split_complexity());
    index_t max_comp_size = (V - 1)/num_thrds + 1;

    /**  get number of components to split  **/
    rV_big = 0; // the number of components to split
    while (rV_big < rV && !is_saturated[rV_big] &&
           first_vertex[rV_big + 1] - first_vertex[rV_big] > max_comp_size){
            rV_big++;
    }

    if (!rV_big){
        rV_new = 0;
        return num_thrds;
    }

    /**  split big components and create balanced component list  **/
    /* the number of resulting new components */
    comp_t rV_new_par = 0; // auxiliary variable for parallel region

    /** there is need to scan all edges involving a given vertex without
     * running through all edges of the graph, so we create the list of
     * 'reverse edges' within each component; to facilitate this, we keep the
     * index of each vertex within its component **/
    index_in_comp = (index_t*) malloc_check(sizeof(index_t)*V);

    #pragma omp parallel for schedule(dynamic) \
        NUM_THREADS(2*E*first_vertex[rV_big]/V, rV_big) reduction(+:rV_new_par)
    for (comp_t rv = 0; rv < rV_big; rv++){
        index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];

        /* adjust maximum component size */
        index_t n = (comp_size - 1)/max_comp_size + 1;
        index_t max_comp_size_rv = (comp_size - 1)/n + 1;

        /* cleanup assigned components */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            comp_assign[comp_list[i]] = NOT_ASSIGNED;
        }

        /* get reverse binding edges for breadth-first search */
        index_t *first_edge_r, *adj_vertices_r;
        get_bind_reverse_edges(rv, first_edge_r, adj_vertices_r);

        /* auxiliary component list for reordering vertices */
        index_t* tmp_comp_list_rv = (index_t*)
            malloc_check(sizeof(index_t)*comp_size);

        /**  compute the new components  **/
        index_t i = 0, j = 0;
        for (index_t k = first_vertex[rv]; k < first_vertex[rv + 1]; k++){
            index_t u = comp_list[k];
            if (comp_assign[u] != NOT_ASSIGNED){ continue; }
            comp_assign[u] = ASSIGNED_ROOT; // flag a component's root
            /* put in connected components list */
            tmp_comp_list_rv[j++] = u;
            index_t size = 1;
            while (i < j){ /* breadth-first search up to max component size */
                index_t v = tmp_comp_list_rv[i++];
                /* add neighbors to the connected component list */
                index_t e = first_edge[v];
                index_t l = index_in_comp[v];
                const index_t* adj_vert = adj_vertices;
                while (adj_vert == adj_vertices || e < first_edge_r[l + 1]){
                    if (adj_vert == adj_vertices){
                        if (e == first_edge[v + 1]){
                            e = first_edge_r[l];
                            adj_vert = adj_vertices_r;
                            continue;
                        }else if (!is_bind(e)){
                            e++; continue;
                        }
                    }
                    index_t w = adj_vert[e];
                    if (comp_assign[w] == NOT_ASSIGNED){
                        comp_assign[w] = ASSIGNED;
                        tmp_comp_list_rv[j++] = w;
                        size++;
                        if (size == max_comp_size_rv){
                            i = j; // incorporate the whole queue
                            break;
                        }
                    }
                    e++;
                }
            } /* the current new component is complete */
            rV_new_par++;
        }

        free(first_edge_r); free(adj_vertices_r);

        index_t* comp_list_rv = comp_list + first_vertex[rv];
        for (index_t i = 0; i < comp_size; i++){
            comp_list_rv[i] = tmp_comp_list_rv[i];
        }

        free(tmp_comp_list_rv);
    }

    rV_new = rV_new_par;

    free(index_in_comp); index_in_comp = nullptr;

    comp_t rV_dif = rV_new - rV_big;

    if ((index_t) rV + rV_dif > MAX_NUM_COMP){
        cerr << "Cut-pursuit: number of balanced components (" <<
            (index_t) rV + rV_dif << ") greater "
            << "than can be represented by comp_t (" << MAX_NUM_COMP << ")"
            << endl;
        exit(EXIT_FAILURE);
    }

    /**  first vertices of balanced components  **/
    comp_t rV_bal = rV + rV_dif;
    index_t* first_vertex_bal = (index_t*)
        malloc_check(sizeof(index_t)*(rV_bal + ONE));

    /* new components first vertices, and assignments for later */
    comp_t rv_new = (comp_t) -1;
    for (index_t i = 0; i < first_vertex[rV_big]; i++){
        index_t v = comp_list[i];
        if (comp_assign[v] == ASSIGNED_ROOT){ first_vertex_bal[++rv_new] = i; }
        comp_assign[v] = rv_new;
    }

    /* add the small components first vertices */
    for (comp_t rv = rV_big; rv < rV + ONE; rv++){
        first_vertex_bal[rv + rV_dif] = first_vertex[rv];
    }

    /**  set parallel cut separation on edges between new components  **/
    #pragma omp parallel for schedule(static) \
        NUM_THREADS(E*first_vertex_bal[rV_new]/V, rV_new)
    for (comp_t rv_new = 0; rv_new < rV_new; rv_new++){
        for (index_t i = first_vertex_bal[rv_new];
             i < first_vertex_bal[rv_new + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_bind(e) && rv_new != comp_assign[adj_vertices[e]]){
                    edge_status[e] = PAR_SEP;
                }
            }
        }
    }

    /**  duplicate component values and saturation accordingly  **/
    rX = (value_t*) realloc_check(rX, sizeof(value_t)*D*rV_bal);
    is_saturated = (bool*) realloc_check(is_saturated, sizeof(bool)*rV_bal);
    /* small components; in-place, start by the end */
    for (comp_t rv = rV - 1; rv >= rV_big; rv--){ // rVbig > 0
        value_t* rXv = rX + D*rv;
        value_t* rXv_bal = rX + D*(rv + rV_dif);
        for (size_t d = 0; d < D; d++){ rXv_bal[d] = rXv[d]; }
        is_saturated[rv + rV_dif] = is_saturated[rv];
    }
    /* big components; in-place, slightly more complicated */
    rv_new = rV_new - 1;
    for (comp_t rv = rV_big; rv --> 0; ){ // nice trick for unsigned comp_t
        value_t* rXv = rX + D*rv;
        while (rv_new != 0 && first_vertex_bal[rv_new] >= first_vertex[rv]){
            value_t* rXv_bal = rX + D*rv_new;
            for (size_t d = 0; d < D; d++){ rXv_bal[d] = rXv[d]; }
            is_saturated[rv_new] = is_saturated[rv]; // should be false
            rv_new--;
        }
    }

    /**  replace the component list by the balanced one  **/
    /* store info abount big components */
    first_vertex_big = (index_t*) realloc_check(first_vertex,
        sizeof(index_t)*(rV_big + 1));
    first_vertex = first_vertex_bal;
    rV = rV_bal;

    return num_thrds;
}

TPL void CP::revert_balance_parallel_split(comp_t rV_new, comp_t rV_big, 
    index_t* first_vertex_big)
{
    index_t* first_vertex_bal = first_vertex; // make clear which one is which
    comp_t rV_dif = rV_new - rV_big;
    comp_t rV_ini = rV - rV_dif;

    /**  remove duplicated component values and aggregate saturation **/
    /* big components */
    comp_t rv_new = 0;
    for (comp_t rv = 0; rv < rV_big; rv++){
        value_t* rXv = rX + D*rv;
        value_t* rXv_bal = rX + D*rv_new;
        for (size_t d = 0; d < D; d++){ rXv[d] = rXv_bal[d]; }

        /* each new component which has not been cut is declared saturated;
         * however, if any new component from the same original large component
         * has been cut, the original large component is not saturated */
        bool saturation = true;
        while (first_vertex_bal[rv_new] < first_vertex_big[rv + 1]){
            saturation = saturation && is_saturated[rv_new];
            rv_new++;
        }
        is_saturated[rv] = saturation;
    }
    /* small components */
    for (comp_t rv = rV_big; rv < rV_ini; rv++){
        value_t* rXv = rX + D*rv;
        value_t* rXv_bal = rX + D*(rv + rV_dif);
        for (size_t d = 0; d < D; d++){ rXv[d] = rXv_bal[d]; }
        is_saturated[rv] = is_saturated[rv + rV_dif];
    }
    rX = (value_t*) realloc_check(rX, sizeof(value_t)*D*rV_ini);
    is_saturated = (bool*) realloc_check(is_saturated, sizeof(bool)*rV_ini);

    /**  revert to initial component list  **/
    /* big components */
    for (comp_t rv = 0; rv < rV_big; rv++){
        first_vertex[rv] = first_vertex_big[rv];
    }
    /* small components; in-place */
    for (comp_t rv = rV_big; rv <= rV_ini; rv++){
        first_vertex[rv] = first_vertex[rv + rV_dif];
    }
    first_vertex = (index_t*) realloc_check(first_vertex,
        sizeof(index_t)*(rV_ini + 1));
    free(first_vertex_big);
    rV = rV_ini;
}

TPL index_t CP::split()
{
    index_t activation = 0;

    comp_t rV_new, rV_big;
    index_t* first_vertex_big;
    int num_thrds = balance_parallel_split(rV_new, rV_big, first_vertex_big);

    /* components are processed in parallel but graph structure specifies edges
     * ends with global indexing; the following table enables constant time
     * conversion to indexing within components */
    index_in_comp = (index_t*) malloc_check(sizeof(index_t)*V);

    #pragma omp parallel for schedule(dynamic) num_threads(num_thrds) \
        reduction(+:activation)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated[rv]){ continue; }

        /**  build flow graph structure  **/
        /* set indexing within component and get number of binding edge */
        index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
        const index_t* comp_list_rv = comp_list + first_vertex[rv];
        index_t number_of_edges = 0;
        for (index_t i = 0; i < comp_size; i++){
            index_t v = comp_list_rv[i];
            index_in_comp[v] = i;
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_bind(e)){ number_of_edges++; }
            }
        }
        /* build flow graph structure and set edges */
        Maxflow<index_t, real_t>* maxflow = new Maxflow<index_t, real_t>
            (comp_size, number_of_edges);
        for (index_t i = 0; i < comp_size; i++){
            index_t v = comp_list_rv[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                index_t j = index_in_comp[adj_vertices[e]];
                if (is_bind(e)){ maxflow->add_edge(i, j); }
            }
        }
        
        /**  set capacities and compute maximum flow  **/
        split_component(rv, maxflow);

        /**  activate edges accordingly  **/
        index_t rv_activation = 0;
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_t l = label_assign[v];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_bind(e) && l != label_assign[adj_vertices[e]]){
                    cut(e);
                    rv_activation++;
                }
            }
        }

        is_saturated[rv] = rv_activation == 0;
        activation += rv_activation;

        delete maxflow;
    }

    free(index_in_comp); index_in_comp = nullptr;

    if (rV_new != rV_big){
        activation += remove_parallel_separations(rV_new);

        revert_balance_parallel_split(rV_new, rV_big, first_vertex_big);
    }

    /* reconstruct components assignment */
    #pragma omp parallel for schedule(static) NUM_THREADS(V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            comp_assign[comp_list[i]] = rv;
        }
    }

    return activation;
}

TPL void CP::merge_components(comp_t& ru, comp_t& rv)
{
    /* ensure the smallest component will be the root of the merge chain */
    if (ru > rv){ comp_t tmp = ru; ru = rv; rv = tmp; }
    /* link both chains; update leaf of the merge chain; update root info */
    merge_chains_next[merge_chains_leaf[ru]] = rv;
    merge_chains_leaf[ru] = merge_chains_leaf[rv];
    merge_chains_root[rv] = merge_chains_root[merge_chains_leaf[rv]] = ru;
    /* saturation considerations are taken care of in merge method */
}

TPL index_t CP::merge()
{
    /**  create the chains representing the merged components  **/
    merge_chains_root = (comp_t*) malloc_check(sizeof(comp_t)*rV); 
    merge_chains_next = (comp_t*) malloc_check(sizeof(comp_t)*rV);
    merge_chains_leaf = (comp_t*) malloc_check(sizeof(comp_t)*rV);
    for (comp_t rv = 0; rv < rV; rv++){
        merge_chains_root[rv] = chain_end();
        merge_chains_next[rv] = chain_end();
        merge_chains_leaf[rv] = rv;
    }
    comp_t merge_count = compute_merge_chains();

    /**  at this point, three different component assignments exists:
     **  the one from previous iteration (in last_comp_assign),
     **  the current one after the split (in comp_assign), and
     **  the final one after the merge (to be computed now)  **/

    /**  recompute saturation: compare previous iterate and final assignment,
     **  and flag nonevolving components as saturated  **/
    if (!last_rV){ /* first iteration, no previous assignment available */
        for (comp_t rv = 0; rv < rV; rv++){ is_saturated[rv] = false; }
    }else{
        /* a previous component is flagged nonevolving if it can be assigned a
         * unique final component */
        /* we can reuse storage since for now last_rV <= rV */
        comp_t* saturation_flag = merge_chains_leaf;
        for (comp_t last_rv = 0; last_rv < last_rV; last_rv++){
            saturation_flag[last_rv] = NOT_ASSIGNED;
        }
        /* run along each final component, from their root */
        for (comp_t ru = 0; ru < rV; ru++){
            if (merge_chains_root[ru] != chain_end()){ continue; }
            comp_t last_ru = last_comp_assign[comp_list[first_vertex[ru]]];
            if (saturation_flag[last_ru] == NOT_ASSIGNED){
                saturation_flag[last_ru] = ASSIGNED;
            }else{ /* was already assigned another final component */
                saturation_flag[last_ru] = NOT_SATURATED;
            }
            /* run along the merge chain */
            comp_t rv = ru; 
            while (rv != chain_end()){
                comp_t last_rv = last_comp_assign[comp_list[first_vertex[rv]]];
                if (last_ru != last_rv){ /* previous components do not agree */
                    saturation_flag[last_ru] = saturation_flag[last_rv] =
                        NOT_SATURATED;
                }
                rv = merge_chains_next[rv];
            }
        }
        /* resulting saturation for each final component */
        for (comp_t rv = 0; rv < rV; rv++){
            if (merge_chains_root[rv] != chain_end()){ continue; }
            comp_t last_rv = last_comp_assign[comp_list[first_vertex[rv]]];
            is_saturated[rv] = saturation_flag[last_rv] != NOT_SATURATED;
        }
    }
    free(merge_chains_leaf); // also storage of saturation_flag

    /**  if no merge take place, no update needed  **/
    if (!merge_count){
        free(merge_chains_root);
        free(merge_chains_next);
        return 0;
    }

    /**  construct the final component lists in temporary storage, and update
     **  components saturation, values and first vertex indices in-place  **/
    saturated_comp = 0;
    saturated_vert = 0;

    /* auxiliary components lists */
    index_t* tmp_comp_list = (index_t*) malloc_check(sizeof(index_t)*V);

    comp_t rn = 0; // component number
    index_t i = 0; // index in the final comp_list
    /* each current component is assigned its final component;
     * this can use the same storage as merge chains root, because the only
     * required information is to flag roots (no need to get back to roots),
     * and roots are processed before getting assigned a final component */
    comp_t* final_comp = merge_chains_root;
    for (comp_t ru = 0; ru < rV; ru++){
        if (merge_chains_root[ru] != chain_end()){ continue; }
        /**  ru is a root, create the corresponding final component  **/
        /* copy component value and saturation;
         * can be done in-place because rn <= ru guaranteed */
        const value_t* rXu = rX + D*ru;
        value_t* rXn = rX + D*rn;
        for (size_t d = 0; d < D; d++){ rXn[d] = rXu[d]; }
        if ((is_saturated[rn] = is_saturated[ru])){ saturated_comp++; }
        /* run along the merge chain */
        index_t first = i; // holds index of first vertex of the component
        comp_t rv = ru;
        while (rv != chain_end()){
            final_comp[rv] = rn;
            /* assign all vertices to final component */ 
            for (index_t j = first_vertex[rv]; j < first_vertex[rv + 1]; j++){
                tmp_comp_list[i++] = comp_list[j];
            }
            if (is_saturated[rn]){
                saturated_vert += first_vertex[rv + 1] - first_vertex[rv];
            }
            rv = merge_chains_next[rv];
        }
        /* the root of each chain is the smallest component in the chain, and
         * the current components are visited in increasing order, so now that
         * 'rn' final components have been constructed, at least the first 'rn'
         * current components have been copied, hence 'first_vertex' will not
         * be accessed before position 'rn' anymore; thus modify in-place */
        first_vertex[rn++] = first;
    }

    free(merge_chains_next);

    /* finalize and shrink arrays to fit the reduced number of components */
    first_vertex[rV = rn] = V;
    first_vertex = (index_t*) realloc_check(first_vertex,
        sizeof(index_t)*(rV + 1));
    rX = (value_t*) realloc_check(rX, sizeof(value_t)*D*rV);
    is_saturated = (bool*) realloc_check(is_saturated, sizeof(bool)*rV); 


    /* update components assignments */
    for (index_t v = 0; v < V; v++){ 
        comp_list[v] = tmp_comp_list[v];
        comp_assign[v] = final_comp[comp_assign[v]];
    }

    free(tmp_comp_list);

    /* deactivate edges between merged components */
    index_t deactivation = 0;
    index_t desaturated_vert = 0;
    index_t desaturated_comp = 0;
    #pragma omp parallel for schedule(static) NUM_THREADS(E, rV) \
        reduction(+:deactivation, desaturated_vert, desaturated_comp)
    for (comp_t rv = 0; rv < rV; rv++){
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_t last_rv = last_comp_assign[v];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_bind(e)){ continue; }
                if (rv != comp_assign[adj_vertices[e]]){ continue; }
                if (balance_par_split){
                /* if a large component has been artificially split for
                 * parallel balancing, parallel separation edges must be set
                 * inactive, but edges which have been cut in parallel
                 * might not properly form new components yet and must be
                 * kept for later; the component must also be desaturated */
                    if (last_rv == last_comp_assign[adj_vertices[e]]){ 
                        if (is_cut(e)){ continue; } // do not bind
                        if (is_saturated[rv]){
                            is_saturated[rv] = false;
                            desaturated_comp++;
                            desaturated_vert += first_vertex[rv + 1]
                                - first_vertex[rv];
                        }
                    }
                }
                bind(e);
                deactivation++;
            }
        }
    }
    saturated_comp -= desaturated_comp;
    saturated_vert -= desaturated_vert;

    /* update corresponding reduced edges and weights in-place;
     * some edges will appear several times in the list, important thing is
     * that the corresponding weights sum up to the right quantity;
     * note that rE is thus an upper bound of the actual number of edges */
    index_t final_re = 0;
    for (index_t re = 0; re < rE; re++){
        comp_t ru = final_comp[reduced_edges[TWO*re]];
        comp_t rv = final_comp[reduced_edges[TWO*re + 1]];
        if (ru != rv){
            reduced_edges[TWO*final_re] = ru;
            reduced_edges[TWO*final_re + 1] = rv;
            reduced_edge_weights[final_re] = reduced_edge_weights[re];
            final_re++;
        }
    }

    free(merge_chains_root); // also storage of final_comp

    rE = final_re;
    reduced_edges = (comp_t*) realloc_check(reduced_edges,
            sizeof(comp_t)*2*rE);
    reduced_edge_weights = (real_t*) realloc_check(reduced_edge_weights,
            sizeof(real_t)*rE);

    return deactivation;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp<float, int32_t, int16_t>;
template class Cp<double, int32_t, int16_t>;
template class Cp<float, int32_t, int32_t>;
template class Cp<double, int32_t, int32_t>;
#else
template class Cp<float, uint32_t, uint16_t>;
template class Cp<double, uint32_t, uint16_t>;
template class Cp<float, uint32_t, uint32_t>;
template class Cp<double, uint32_t, uint32_t>;
#endif
