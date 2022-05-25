/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include "cut_pursuit_d0.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define TWO ((size_t) 2) // avoid overflows
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
/* special flag (no component can have this identifier) */
#define MERGE_INIT (std::numeric_limits<comp_t>::max())

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP_D0 Cp_d0<real_t, index_t, comp_t, value_t>

using namespace std;

TPL CP_D0::Cp_d0(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D)
    : Cp<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D),
      accepted_merge(&reserved_merge_info)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "Cut-pursuit d0: real_t must satisfy IEEE 754.");

    K = 2;
    split_iter_num = 2;
    split_damp_ratio = ONE;
}

TPL void CP_D0::set_split_param(comp_t K, int split_iter_num,
    real_t split_damp_ratio)
{
    if (split_iter_num < 1){
        cerr << "Cut-pursuit d0: there must be at least one iteration in the "
            "split (" << split_iter_num << " specified)." << endl;
        exit(EXIT_FAILURE);
    }

    if (K < 2){
        cerr << "Cut-pursuit d0: there must be at least two alternative values"
            "in the split (" << K << " specified)." << endl;
        exit(EXIT_FAILURE);
    }

    if (split_damp_ratio <= 0 || split_damp_ratio > ONE){
        cerr << "Cut-pursuit d0: split damping ratio must be between zero "
            "excluded and one included (" << split_damp_ratio << " specified)."
            << endl;
        exit(EXIT_FAILURE);
    }

    this->K = K;
    this->split_iter_num = split_iter_num;
    this->split_damp_ratio = split_damp_ratio;
}

TPL real_t CP_D0::compute_graph_d0()
{
    real_t weighted_contour_length = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(rE) \
        reduction(+:weighted_contour_length)
    for (index_t re = 0; re < rE; re++){
        weighted_contour_length += reduced_edge_weights[re];
    }
    return weighted_contour_length;
}

TPL real_t CP_D0::compute_f()
{
    real_t f = ZERO;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(D*V, rV) \
        reduction(+:f)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
            f += fv(comp_list[v], rXv);
        }
    }
    return f;
}

TPL real_t CP_D0::compute_objective()
{ return compute_f() + compute_graph_d0(); } // f(x) + ||x||_d0

TPL uintmax_t CP_D0::split_complexity()
{
    uintmax_t complexity = maxflow_complexity(); // graph cut
    complexity += D*V; // account for distance difference and final labeling
    complexity += E; // edges capacities
    if (K > 2){ complexity *= K; } // K alternative labels
    complexity *= split_iter_num; // repeated
    complexity += split_values_complexity(); // init and update
    return complexity*(V - saturated_vert)/V; // account saturation linearly
}

TPL void CP_D0::split_component(comp_t rv, Maxflow<index_t, real_t>* maxflow)
{
    value_t* altX = (value_t*) malloc_check(sizeof(value_t)*D*K);

    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
    const index_t* comp_list_rv = comp_list + first_vertex[rv];

    real_t damping = split_damp_ratio;
    for (int split_it = 0; split_it < split_iter_num; split_it++){
        damping += (ONE - split_damp_ratio)/split_iter_num;

        if (split_it == 0){ init_split_values(rv, altX); }
        else{ update_split_values(rv, altX); }

        bool no_reassignment = true;

        if (K == 2){ /* one graph cut is enough */
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                /* unary cost for choosing the second alternative */
                maxflow->terminal_capacity(i) = fv(v, altX + D) - fv(v, altX);
            }

            /* set d0 edge capacities within each component */
            index_t e_in_comp = 0;
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                    if (is_bind(e)){
                        real_t cap = damping*EDGE_WEIGHTS_(e);
                        maxflow->set_edge_capacities(e_in_comp++, cap, cap);
                    }
                }
            }

            /* find min cut and set assignment accordingly */
            maxflow->maxflow();

            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                if (maxflow->is_sink(i) != label_assign[v]){
                    label_assign[v] = maxflow->is_sink(i);
                    no_reassignment = false;
                }
            }

        }else{ /* iterate over all K alternative values */
            for (comp_t k = 0; k < K; k++){
    
            /* check if alternative k has still vertices assigned to it */
            if (!is_split_value(altX[D*k])){ continue; }

            /* set the source/sink capacities */
            bool all_assigned_k = true;
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                comp_t l = label_assign[v];
                /* unary cost for changing current value to k-th value */
                if (l == k){
                    maxflow->terminal_capacity(i) = ZERO;
                }else{
                    maxflow->terminal_capacity(i) = fv(v, altX + D*k) -
                        fv(v, altX + D*l);
                    all_assigned_k = false;
                }
            }
            if (all_assigned_k){ continue; }

            /* set d0 edge capacities within each component */
            index_t e_in_comp = 0;
            for (index_t i = 0; i < comp_size; i++){
                index_t u = comp_list_rv[i];
                comp_t lu = label_assign[u];
                for (index_t e = first_edge[u]; e < first_edge[u + 1]; e++){
                    if (!is_bind(e)){ continue; }
                    index_t v = adj_vertices[e];
                    comp_t lv = label_assign[v];
                    /* horizontal and source/sink capacities are modified 
                     * according to Kolmogorov & Zabih (2004); in their
                     * notations, functional E(u,v) is decomposed as
                     *
                     * E(0,0) | E(0,1)    A | B
                     * --------------- = -------
                     * E(1,0) | E(1,1)    C | D
                     *                         0 | 0      0 | D-C    0 |B+C-A-D
                     *                 = A + --------- + -------- + -----------
                     *                       C-A | C-A    0 | D-C    0 |   0
                     *
                     *            constant +      unary terms     + binary term
                     */
                    /* A = E(0,0) is the cost of the current assignment */
                    real_t A = lu == lv ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* B = E(0,1) is the cost of changing lv to k */
                    real_t B = lu == k ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* C = E(1,0) is the cost of changing lu to k */
                    real_t C = lv == k ? ZERO : damping*EDGE_WEIGHTS_(e);
                    /* D = E(1,1) = 0 is for changing both lu, lv to k */
                    /* set weights in accordance with orientation u -> v */
                    maxflow->terminal_capacity(i) += C - A;
                    maxflow->terminal_capacity(index_in_comp[v]) -= C;
                    maxflow->set_edge_capacities(e_in_comp++, B + C - A, ZERO);
                }
            }

            /* find min cut and update assignment accordingly */
            maxflow->maxflow();

            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                if (maxflow->is_sink(i) && label_assign[v] != k){
                    label_assign[v] = k;
                    no_reassignment = false;
                }
            }

            } // end for k
        } // end if K == 2

        if (no_reassignment){ break; }

    } // end for split_it

    free(altX);
}


TPL index_t CP_D0::remove_parallel_separations(comp_t rV_new)
{
    index_t activation = 0;

    /* reconstruct component assignment (only on new components) */
    #pragma omp parallel for schedule(static) \
        NUM_THREADS(first_vertex[rV_new], rV_new)
    for (comp_t rv_new = 0; rv_new < rV_new; rv_new++){
        for (index_t i = first_vertex[rv_new]; i < first_vertex[rv_new + 1];
            i++){
            comp_assign[comp_list[i]] = rv_new;
        }
    }

    /* parallel separation edges are kept if at least one end vertex
     * belongs to a nonsaturated component; they should be deactivated
     * at merge step */
    #pragma omp parallel for schedule(static) reduction(+:activation) \
        NUM_THREADS(E*first_vertex[rV_new]/V, rV_new)
    for (comp_t rv_new = 0; rv_new < rV_new; rv_new++){
        const bool sat = is_saturated[rv_new];
        for (index_t i = first_vertex[rv_new]; i < first_vertex[rv_new + 1];
            i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_par_sep(e)){
                    if (sat && is_saturated[comp_assign[adj_vertices[e]]]){
                        bind(e);
                    }else{
                        activation++;
                    }
                }
            }
        }
    }

    return activation;
}

TPL CP_D0::Merge_info::Merge_info(size_t D)
{ value = (value_t*) malloc_check(sizeof(value_t)*D); }

TPL CP_D0::Merge_info::~Merge_info()
{ free(value); }

TPL void CP_D0::delete_merge_candidate(index_t re)
{
    delete merge_info_list[re];
    merge_info_list[re] = accepted_merge;
}

TPL void CP_D0::select_best_merge_candidate(index_t re, real_t* best_gain,
    index_t* best_edge)
{
    if (merge_info_list[re] && merge_info_list[re]->gain > *best_gain){
            *best_gain = merge_info_list[re]->gain;
            *best_edge = re;
    }
}

TPL void CP_D0::accept_merge_candidate(index_t re, comp_t& ru, comp_t& rv)
{
    merge_components(ru, rv); // ru now the root of the merge chain
    value_t* rXu = rX + D*ru;
    for (size_t d = 0; d < D; d++){ rXu[d] = merge_info_list[re]->value[d]; }
}

TPL comp_t CP_D0::compute_merge_chains()
{
    comp_t merge_count = 0;
   
    merge_info_list = (Merge_info**) malloc_check(sizeof(Merge_info*)*rE);
    for (index_t re = 0; re < rE; re++){ merge_info_list[re] = nullptr; }

    real_t* best_par_gains =
        (real_t*) malloc_check(sizeof(real_t)*omp_get_num_procs());
    index_t* best_par_edges = 
        (index_t*) malloc_check(sizeof(index_t)*omp_get_num_procs());

    comp_t last_merge_root = MERGE_INIT;

    while (true){
 
        /**  update merge information in parallel  **/
        int num_par_thrds = last_merge_root == MERGE_INIT ?
            compute_num_threads(update_merge_complexity()) :
            /* expected fraction of merge candidates to update is the total
             * number of edges divided by the expected number of edges linking
             * to the last merged component; in turn, this is estimated as
             * twice the number of edges divided by the number of components */
            compute_num_threads(update_merge_complexity()/rV*2);

        for (int thrd_num = 0; thrd_num < num_par_thrds; thrd_num++){
            best_par_gains[thrd_num] = -real_inf();
        }

        /* differences between threads is small: using static schedule */
        #pragma omp parallel for schedule(static) num_threads(num_par_thrds)
        for (index_t re = 0; re < rE; re++){
            if (merge_info_list[re] == accepted_merge){ continue; }
            comp_t ru = reduced_edges[TWO*re];
            comp_t rv = reduced_edges[TWO*re + 1];

            if (last_merge_root != MERGE_INIT){
                /* the roots of their respective chains might have changed */
                ru = get_merge_chain_root(ru);
                rv = get_merge_chain_root(rv);
                /* check if none of them is concerned by the last merge */
                if (last_merge_root != ru && last_merge_root != rv){
                    select_best_merge_candidate(re,
                        best_par_gains + omp_get_thread_num(),
                        best_par_edges + omp_get_thread_num());
                    continue;
                }
            }

            if (ru == rv){ /* already merged */
                delete_merge_candidate(re);
            }else{ /* update information */
                update_merge_candidate(re, ru, rv);
                select_best_merge_candidate(re,
                    best_par_gains + omp_get_thread_num(),
                    best_par_edges + omp_get_thread_num());
            }
        } // end for candidates in parallel

        /**  select best candidate among parallel threads  **/
        real_t best_gain = best_par_gains[0];
        index_t best_edge = best_par_edges[0];
        for (int thrd_num = 1; thrd_num < num_par_thrds; thrd_num++){
            if (best_gain < best_par_gains[thrd_num]){
                best_gain = best_par_gains[thrd_num];
                best_edge = best_par_edges[thrd_num];
            }
        }

        /**  merge best candidate if best gain is positive  **/
        /* we allow for negative gains, as long as its not negative infinity */
        if (best_gain > -real_inf()){
            comp_t ru = get_merge_chain_root(reduced_edges[2*best_edge]);
            comp_t rv = get_merge_chain_root(reduced_edges[2*best_edge + 1]);
            accept_merge_candidate(best_edge, ru, rv); // ru now the root
            delete_merge_candidate(best_edge);
            merge_count++;
            last_merge_root = ru;
        }else{
            break;
        }
   
    } // end merge loop

    free(best_par_gains);
    free(best_par_edges);
    free(merge_info_list); // all merge info must have been deleted

    return merge_count;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    template class Cp_d0<float, int32_t, int16_t>;
    template class Cp_d0<double, int32_t, int16_t>;
    template class Cp_d0<float, int32_t, int32_t>;
    template class Cp_d0<double, int32_t, int32_t>;
#else
    template class Cp_d0<float, uint32_t, uint16_t>;
    template class Cp_d0<double, uint32_t, uint16_t>;
    template class Cp_d0<float, uint32_t, uint32_t>;
    template class Cp_d0<double, uint32_t, uint32_t>;
#endif
