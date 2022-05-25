/*=============================================================================
 * Derived class for cut-pursuit algorithm with d1 (total variation) 
 * penalization, with a quadratic functional, l1 penalization and box
 * constraints:
 *
 * minimize functional over a graph G = (V, E)
 *
 *        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)
 *
 * where y in R^N, x in R^V, A in R^{N-by-|V|}, yl1 in R^V
 *      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
 *      ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
 * and the convex indicator
 *      i_[m,M](x) = infinity any x_v < m_v or x_v > M_v
 *                 = 0 otherwise;
 *
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * It is easy to introduce a SDP metric weighting the squared l2-norm
 * between y and A x. Indeed, if M is the matrix of such a SDP metric,
 *   ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
 * Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
 * Moreover, when A is the identity and M is diagonal (weighted square l2
 * distance between x and y), one should call on the precomposed version 
 * (N set to Gram_diag(), see below) with Y <- DDy = My and A <- D2 = M.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
 * Nonsmooth Functionals with Graph Total Variation, International Conference
 * on Machine Learning, PMLR, 2018, 80, 4244-4253
 *
 * H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
 * Inclusion and Convex Optimization Optimization Letters, 2018, 1-24
 *
 * Hugo Raguet 2018, 2020
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
class Cp_d1_ql1b : public Cp_d1<real_t, index_t, comp_t>
{
private:
    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::dif_tol;

public:
    /**  constructor, destructor  **/

    /* only creates BK graph structure and assign Y */
    Cp_d1_ql1b(index_t V, index_t E, const index_t* first_edge,
        const index_t* adj_vertices);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); IT DOES FREE THE REST
     * (components assignment and reduced problem elements, etc.), but this can
     * be prevented by getting the corresponding pointer member and setting it
     * to null beforehand */
    ~Cp_d1_ql1b();

    #if defined _OPENMP && _OPENMP < 200805
    /* use of unsigned counter in parallel loops requires OpenMP 3.0;
     * although published in 2008, MSVC still does not support it as of 2020 */
    typedef long int matrix_index_t;
    #else
    typedef size_t matrix_index_t;
    #endif

    /* flag Gram matrices */
    static matrix_index_t Gram_full() { return 0; }
    static matrix_index_t Gram_diag() { return -1; }
    static bool is_Gram(matrix_index_t N)
        { return N == Gram_full() || N == Gram_diag(); }

    /* set the quadratic part, see members Y, N, A for details;
     * set Y to a null pointer for all zeros;
     * set A to a null pointer for identity matrix (set a to nonzero), or for
     * no quadratic part (set a to zero);
     * for a general scalar matrix, use the identity (A null, a zero) and scale
     * observations and penalizations accordingly */
    void set_quadratic(const real_t* Y, matrix_index_t N,
        const real_t* A, real_t a = 1.0);

    /* overload for identity matrix */
    void set_quadratic(const real_t* Y)
        { set_quadratic(Y, Gram_diag(), nullptr); }

    /* set l1_weights null for homogeneously equal to homo_l1_weight */
    void set_l1(const real_t* l1_weights = nullptr,
        real_t homo_l1_weight = 0.0, const real_t* Yl1 = nullptr);

    /* make real infinity publicly available */ 
    static real_t real_inf(){ return Cp<real_t, index_t, comp_t>::real_inf(); }
    
    /* set bounds *_bnd to null for homogeneously equal to homo_*_bnd */
    void set_bounds(
        const real_t* low_bnd = nullptr, real_t homo_low_bnd = -real_inf(),
        const real_t* upp_bnd = nullptr, real_t homo_upp_bnd = real_inf());

    void set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
        int it_max, real_t dif_tol);

    /* overload for default dif_tol parameter */
    void set_pfdr_param(real_t rho = 1.0, real_t cond_min = 1e-3,
        real_t dif_rcd = 0.0, int it_max = 1e4)
    { set_pfdr_param(rho, cond_min, dif_rcd, it_max, 1e-3*dif_tol); }

private:

    /**  main problem  **/

    /* quadratic problem */

    matrix_index_t N; /* number of observations;
     * if zero (function Gram_full()), matricial information is precomputed, 
     * that is, argument A is actually (A^t A), and argument Y is (A^t Y);
     * if negative one (or maximum value representable by matrix_index_t if
     * unsigned type, macro Gram_diag()), A is a diagonal matrix and only the
     * diagonal of (A^t A) = A^2 is given */
    
    const real_t* A; /* linear operator;
     * if N is positive, N-by-V array, column major format;
     * if N is zero (function Gram_full()), matrix (A^t A), V-by-V array,
     * column major format;
     * if N is negative one (function Gram_diag()), diagonal of (A^t A) = A^2,
     * array of length V, or null pointer for identity matrix (a = 1) or no
     * quadratic part (a = 0) */
    real_t a; 

    const real_t* Y; /* if N is positive, observations, array of length N;
     * otherwise, correlation of A with the observations (A^t Y), array of
     * length V; set to null for all zero */

    real_t *R; // residual, array of length N, used only if N is positive

    /* regularizations */

    /* observations for l1 fidelity, array of length V, set to null for zero */
    const real_t* Yl1;

    /* l1 penalization coefficients;
     * if 'l1_weights' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_d1_weight' */
    const real_t* l1_weights;
    real_t homo_l1_weight;
    /* lower bounds of box constraints;
     * if 'low_bnd' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_low_bnd' */
    const real_t* low_bnd;
    real_t homo_low_bnd;
    /* upper bounds of box constraints;
     * if 'upp_bnd' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_upp_bnd' */
    const real_t* upp_bnd;
    real_t homo_upp_bnd;

    /**  reduced problem  **/
    real_t pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int pfdr_it, pfdr_it_max;

    /**  cut-pursuit steps  **/

    /* split */
    /* rough estimate of the number of operations for split step;
     * useful for estimating the number of parallel threads */
    uintmax_t split_complexity() override;
    real_t* grad; // store gradient of smooth part
    void split_component(comp_t rv, Maxflow<index_t, real_t>* maxflow)
        override;
    index_t split() override; // overload for computing gradient

    /* compute reduced values;
     * NOTA: if Yl1 is not constant, this actually solves only an approximation
     * of the reduced problem, replacing the weighted sum of distances to Yl1
     * by the distance to the weighted median of Yl1 */
    void solve_reduced_problem() override;

    /* relative iterate evolution in l2 norm and components saturation */
    real_t compute_evolution(bool compute_dif) override;

    /* in the precomputed A^t A version, a constant 1/2||Y||^2 in the quadratic
     * part is omited */
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
