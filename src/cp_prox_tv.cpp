/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <cmath>
#include "cp_prox_tv.hpp"
#include "pfdr_d1_ql1b.hpp"
#include "wth_element.hpp"

#define ZERO ((real_t) 0.0) // avoid conversions
#define HALF ((real_t) 0.5) // avoid conversions
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_PROX_TV Cp_prox_tv<real_t, index_t, comp_t>
#define PFDR Pfdr_d1_ql1b<real_t, comp_t>

using namespace std;

TPL CP_PROX_TV::Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices)
    : Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices)
{
    Y = Gd1 = nullptr;

    pfdr_rho = 1.0; pfdr_cond_min = 1e-2; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-3*dif_tol; pfdr_it = pfdr_it_max = 1e4;

    /* it makes sense to consider nonevolving components as saturated */
    monitor_evolution = true;
}

TPL void CP_PROX_TV::set_observation(const real_t* Y)
{
    this->Y = Y;
}

TPL void CP_PROX_TV::set_d1_subgradients(real_t* Gd1)
{
    this->Gd1 = Gd1;
}

TPL void CP_PROX_TV::set_pfdr_param(real_t rho, real_t cond_min,
    real_t dif_rcd, int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_PROX_TV::solve_reduced_problem()
{
    /**  compute reduced matrix  **/
    real_t *rY, *rAA; // reduced observations and matrix
    rY = rAA = nullptr;

    rY = (real_t*) malloc_check(sizeof(real_t)*rV);
    rAA = (real_t*) malloc_check(sizeof(real_t)*rV);

    #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        rY[rv] = ZERO;
        /* run along the component rv */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
            i++){
            rY[rv] += Y[comp_list[i]];
        }
    }
    
    #pragma omp parallel for schedule(static) NUM_THREADS(rV)
    for (comp_t rv = 0; rv < rV; rv++){
        rAA[rv] = first_vertex[rv + 1] - first_vertex[rv];
    }

    if (rV == 1){ /**  single connected component  **/

        *rX = (*rY)/(*rAA);

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        Pfdr_d1_ql1b<real_t, comp_t> *pfdr =
            new Pfdr_d1_ql1b<real_t, comp_t>(rV, rE, reduced_edges);

        pfdr->set_edge_weights(reduced_edge_weights);
        pfdr->set_quadratic(rY, PFDR::Gram_diag(), rAA);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, sqrt(pfdr_it_max),
            pfdr_it_max, verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d
        delete pfdr;

    }

    free(rY); free(rAA);
}

TPL index_t CP_PROX_TV::split()
{
    index_t activation = Cp<real_t, index_t, comp_t>::split();

    return activation;
}

TPL uintmax_t CP_PROX_TV::split_complexity()
{
    uintmax_t complexity = maxflow_complexity(); // graph cut
    complexity += V; // account for gradient and final labeling
    complexity += E; // edges capacities
    return complexity*(V - saturated_vert)/V; // account saturation linearly
}

TPL void CP_PROX_TV::split_component(comp_t rv,
    Maxflow<index_t, real_t>* maxflow)
{
    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
    const index_t* comp_list_rv = comp_list + first_vertex[rv];

    /**  the cut is essentially +1 vs -1
     **  actual derivative value is twice the cut cost plus a constant  */

    real_t rXv = rX[rv];

    /* set gradient of quadratic term on terminal capacities */
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        maxflow->terminal_capacity(i) = rXv - Y[v];
    }

    /* set the d1 edge and terminal capacities */
    index_t e_in_comp = 0;
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_bind(e)){
                maxflow->set_edge_capacities(e_in_comp++, EDGE_WEIGHTS_(e),
                    EDGE_WEIGHTS_(e));
            }else if (is_cut(e)){
                index_t u = adj_vertices[e]; /* edge (v, u): |x_v - x_u| */
                maxflow->terminal_capacity(i) += rXv > rX[comp_assign[u]] ?
                    EDGE_WEIGHTS_(e) : -EDGE_WEIGHTS_(e);
            /* in most cases, both sides of a parallel separation
             * prefer the same descent direction, so no additional capacity;
             * this favors cutting, usually not detrimental to optimality */
            /* }else if (is_par_sep(e)){
                maxflow->terminal_capacity(i) += EDGE_WEIGHTS_(e); */
            }
        }
    }

    /* find min cut */
    maxflow->maxflow();

    /* assign label accordingly and get subgradients from flows */
    e_in_comp = 0;
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        label_assign[v] = maxflow->is_sink(i);
        if (!Gd1){ continue; }
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_bind(e)){
                Gd1[e] = maxflow->get_edge_flow(e_in_comp++, EDGE_WEIGHTS_(e));
            }
        }
    }
}

TPL real_t CP_PROX_TV::compute_evolution(bool compute_dif)
{
    index_t num_ops = compute_dif ? (V - saturated_vert) : saturated_comp;
    real_t dif = ZERO, amp = ZERO;
    /* auxiliary variable for parallel region */
    comp_t saturated_comp_par = 0; 
    index_t saturated_vert_par = 0;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(num_ops, rV) \
        reduction(+:dif, amp, saturated_comp_par, saturated_vert_par)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t rXv = rX[rv];
        if (is_saturated[rv]){
            real_t lrXv = last_rX[
                last_comp_assign[comp_list[first_vertex[rv]]] ];
            real_t rv_dif = abs(rXv - lrXv);
            if (rv_dif > abs(rX[rv])*dif_tol){
                is_saturated[rv] = false;
            }else{
                saturated_comp_par++;
                saturated_vert_par += first_vertex[rv + 1] - first_vertex[rv];
            }
            if (compute_dif){
                dif += rv_dif*rv_dif*(first_vertex[rv + 1] - first_vertex[rv]);
                amp += rXv*rXv*(first_vertex[rv + 1] - first_vertex[rv]);
            }
        }else if (compute_dif){
            for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
                real_t lrXv = last_rX[last_comp_assign[comp_list[v]]];
                dif += (rXv - lrXv)*(rXv - lrXv);
            }
            amp += rXv*rXv*(first_vertex[rv + 1] - first_vertex[rv]);
        }
    }
    saturated_comp = saturated_comp_par;
    saturated_vert = saturated_vert_par;
    if (compute_dif){
        dif = sqrt(dif);
        amp = sqrt(amp);
        return amp > eps ? dif/amp : dif/eps;
    }else{
        return real_inf();
    }
}

TPL real_t CP_PROX_TV::compute_objective()
/* unfortunately, at this point one does not have access to the reduced objects
 * computed in the routine solve_reduced_problem() */
{
    real_t obj = ZERO;

    /* quadratic part up to the constant 1/2||Y||^2 */
    #pragma omp parallel for reduction(+:obj) schedule(dynamic) \
        NUM_THREADS(V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t rAAv = first_vertex[rv + 1] - first_vertex[rv];
        real_t rAYv = ZERO;
        /* run along the component rv */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            rAYv += Y[i];
        }
        obj += rX[rv]*(HALF*rAAv*rX[rv] - rAYv);
    }

    obj += compute_graph_d1(); // ||x||_d1

    return obj;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_prox_tv<double, int32_t, int16_t>;
template class Cp_prox_tv<float, int32_t, int16_t>;
template class Cp_prox_tv<double, int32_t, int32_t>;
template class Cp_prox_tv<float, int32_t, int32_t>;
#else
template class Cp_prox_tv<double, uint32_t, uint16_t>;
template class Cp_prox_tv<float, uint32_t, uint16_t>;
template class Cp_prox_tv<double, uint32_t, uint32_t>;
template class Cp_prox_tv<float, uint32_t, uint32_t>;
#endif
