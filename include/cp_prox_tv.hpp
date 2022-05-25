/*=============================================================================
 * Derived class for cut-pursuit algorithm for the proximal operator of the 
 * total variation, that is square quadratic difference with d1 penalization :
 *
 * minimize functional over a graph G = (V, E)
 *
 *        F(x) = 1/2 ||y - x||^2 + ||x||_d1
 *
 * where x, y in R^V, and
 *      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
 *
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * L. Landrieu and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn 
 * Piecewise Constant Functions on General Weighted Graphs, SIAM Journal on 
 * Imaging Sciences, 2017, 10, 1724-1766
 *
 * Hugo Raguet 2021
 *===========================================================================*/
#pragma once
#include "cut_pursuit_d1.hpp"

/* real_t is the real numeric type, used for the base field and for the
 * objective functional computation;
 * index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
template <typename real_t, typename index_t, typename comp_t>
class Cp_prox_tv : public Cp_d1<real_t, index_t, comp_t>
{
private:
    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::dif_tol;

public:
    /**  constructor, destructor  **/

    /* only creates BK graph structure and assign Y */
    Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
        const index_t* adj_vertices);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); IT DOES FREE THE REST
     * (components assignment and reduced problem elements, etc.), but this can
     * be prevented by getting the corresponding pointer member and setting it
     * to null beforehand */

    /* set the observation, see member Y for details */
    void set_observation(const real_t* Y);

    /* set an array for storing d1 subgradients, see member Gd1 for details */
    void set_d1_subgradients(real_t* Gd1);

    /* make real infinity publicly available */ 
    static real_t real_inf(){ return Cp<real_t, index_t, comp_t>::real_inf(); }
    
    void set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
        int it_max, real_t dif_tol);

    /* overload for default dif_tol parameter */
    void set_pfdr_param(real_t rho = 1.0, real_t cond_min = 1e-3,
        real_t dif_rcd = 0.0, int it_max = 1e4)
    { set_pfdr_param(rho, cond_min, dif_rcd, it_max, 1e-3*dif_tol); }

private:

    /**  main problem  **/

    const real_t* Y; // observations, array of length V

    real_t* Gd1; // subgradients of d1 (total variation penalization)

    /**  reduced problem  **/

    real_t pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int pfdr_it, pfdr_it_max;

    /**  cut-pursuit steps  **/

    /* split */
    /* rough estimate of the number of operations for split step;
     * useful for estimating the number of parallel threads */
    uintmax_t split_complexity() override;
    void split_component(comp_t rv, Maxflow<index_t, real_t>* maxflow)
        override;
    index_t split() override; // overload for computing gradient

    /* compute reduced values */
    void solve_reduced_problem() override;

    /* relative iterate evolution in l2 norm and components saturation */
    real_t compute_evolution(bool compute_dif) override;

    real_t compute_objective() override;

    /**  type resolution for base template class members  **/
    using Cp_d1<real_t, index_t, comp_t>::compute_graph_d1;
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::last_rX;
    using Cp<real_t, index_t, comp_t>::monitor_evolution;
    using Cp<real_t, index_t, comp_t>::is_cut;
    using Cp<real_t, index_t, comp_t>::is_bind;
    using Cp<real_t, index_t, comp_t>::is_par_sep;
    using Cp<real_t, index_t, comp_t>::is_saturated;
    using Cp<real_t, index_t, comp_t>::saturated_comp;
    using Cp<real_t, index_t, comp_t>::saturated_vert;
    using Cp<real_t, index_t, comp_t>::last_comp_assign;
    using Cp<real_t, index_t, comp_t>::maxflow_complexity;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::edge_weights;
    using Cp<real_t, index_t, comp_t>::homo_edge_weight;
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::comp_assign;
    using Cp<real_t, index_t, comp_t>::comp_list;
    using Cp<real_t, index_t, comp_t>::label_assign;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::reduced_edges;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::eps;
    using Cp<real_t, index_t, comp_t>::verbose;
    using Cp<real_t, index_t, comp_t>::malloc_check;
};
