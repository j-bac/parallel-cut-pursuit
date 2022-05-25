/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <random>
#include "cp_kmpp_d0_dist.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define VERT_WEIGHTS_(v) (vert_weights ? vert_weights[(v)] : ONE)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D0_DIST Cp_d0_dist<real_t, index_t, comp_t>

using namespace std;

TPL CP_D0_DIST::Cp_d0_dist(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, const real_t* Y, size_t D)
    : Cp_d0<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D), Y(Y)
{
    vert_weights = coor_weights = nullptr;
    comp_weights = nullptr; 
    kmpp_init_num = 3;
    kmpp_iter_num = 3;

    loss = quadratic_loss();
    fYY = ZERO;
    fXY = real_inf();

    min_comp_weight = ZERO;
}

TPL CP_D0_DIST::~Cp_d0_dist(){ free(comp_weights); }

TPL inline real_t CP_D0_DIST::distance(const real_t* Yv, const real_t* Xv)
{
    real_t dist = 0.0;
    if (loss == quadratic_loss()){
        if (coor_weights){
            for (size_t d = 0; d < D; d++){
                dist += coor_weights[d]*(Yv[d] - Xv[d])*(Yv[d] - Xv[d]);
            }
        }else{
            for (size_t d = 0; d < D; d++){
                dist += (Yv[d] - Xv[d])*(Yv[d] - Xv[d]);
            }
        }
    }else{ // smoothed Kullback-Leibler; just compute cross-entropy here
        const real_t c = ((real_t) 1.0 - loss);
        const real_t q = loss/D;
        if (coor_weights){
            for (size_t d = 0; d < D; d++){
                dist -= coor_weights[d]*(q + c*Yv[d])*log(q + c*Xv[d]);
            }
        }else{
            for (size_t d = 0; d < D; d++){
                dist -= (q + c*Yv[d])*log(q + c*Xv[d]);
            }
        }
    }
    return dist;
}

TPL void CP_D0_DIST::set_loss(real_t loss, const real_t* Y,
    const real_t* vert_weights, const real_t* coor_weights)
{
    if (loss < ZERO || loss > ONE){
        cerr << "Cut-pursuit d0 distance: loss parameter should be between"
            " 0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if (loss == ZERO){ loss = eps; } // avoid singularities
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->vert_weights = vert_weights;
    this->coor_weights = coor_weights; 
    /* recompute the constant dist(Y, Y) if necessary */
    real_t fYY_par = ZERO; // auxiliary variable for parallel region
    if (loss != quadratic_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:fYY_par)
        for (index_t v = 0; v < V; v++){
            const real_t* Yv = Y + D*v;
            fYY_par += VERT_WEIGHTS_(v)*distance(Yv, Yv);
        }
    }
    fYY = fYY_par;
}

TPL void CP_D0_DIST::set_kmpp_param(int kmpp_init_num, int kmpp_iter_num)
{
    this->kmpp_init_num = kmpp_init_num;
    this->kmpp_iter_num = kmpp_iter_num;
}

TPL void CP_D0_DIST::set_min_comp_weight(real_t min_comp_weight)
{
    if (min_comp_weight < ZERO){
        cerr << "Cut-pursuit d0 distance: min component weight parameter "
            "should be positive (" << min_comp_weight << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    this->min_comp_weight = min_comp_weight;
}

TPL real_t CP_D0_DIST::fv(index_t v, const real_t* Xv)
{ return VERT_WEIGHTS_(v)*distance(Y + D*v, Xv); }

TPL real_t CP_D0_DIST::compute_f()
{
    if (fXY == real_inf()){
        fXY = Cp_d0<real_t, index_t, comp_t>::compute_f();
    }
    return fXY - fYY;
}

TPL void CP_D0_DIST::solve_reduced_problem()
{
    free(comp_weights);
    comp_weights = (real_t*) malloc_check(sizeof(real_t)*rV);
    fXY = real_inf(); // rX will change, fXY must be recomputed

    #pragma omp parallel for schedule(static) NUM_THREADS(2*D*V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        comp_weights[rv] = ZERO;
        for (size_t d = 0; d < D; d++){ rXv[d] = ZERO; }
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_weights[rv] += VERT_WEIGHTS_(v);
            const real_t* Yv = Y + D*v;
            for (size_t d = 0; d < D; d++){ rXv[d] += VERT_WEIGHTS_(v)*Yv[d]; }
        }
        if (comp_weights[rv]){
            for (size_t d = 0; d < D; d++){ rXv[d] /= comp_weights[rv]; }
        } /* maybe one should raise an exception for zero weight component */
    }
}

TPL uintmax_t CP_D0_DIST::split_values_complexity()
{
    uintmax_t complexity = (uintmax_t) K*V; // draw initialization
    complexity += D*K*V*2*kmpp_iter_num; // k-means
    complexity *= kmpp_init_num; // repetition
    complexity += D*K*V*(split_iter_num - 1); // update centroids
    return complexity;
}

TPL void CP_D0_DIST::init_split_values(comp_t rv, real_t* altX)
{
    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];
    const index_t* comp_list_rv = comp_list + first_vertex[rv];

    /* distance map and random device for k-means++ */
    real_t* nearest_dist = (real_t*) malloc_check(sizeof(real_t)*comp_size);
    default_random_engine rand_gen; // default seed also enough for our purpose

    /* current centroids, min sum of distances and corresponding assignment */
    real_t* centroids = (real_t*) malloc_check(sizeof(real_t)*D*K);
    real_t min_sum_dist = real_inf();
    comp_t* best_assign = (comp_t*) malloc_check(sizeof(comp_t)*comp_size);

    /* store centroids entropy for Kullback-Leibler divergence */
    real_t* bottom_dist = loss == quadratic_loss() ?
        nullptr : (real_t*) malloc_check(sizeof(real_t)*K);

    /**  kmeans ++  **/
    for (int kmpp_init = 0; kmpp_init < kmpp_init_num; kmpp_init++){

        /**  initialization  **/ 
        for (comp_t k = 0; k < K; k++){
            index_t rand_i;
            if (k == 0){
                uniform_int_distribution<index_t> unif_distr(0, comp_size - 1);
                rand_i = unif_distr(rand_gen);
            }else{
                for (index_t i = 0; i < comp_size; i++){
                    index_t v = comp_list_rv[i];
                    nearest_dist[i] = real_inf();
                    for (comp_t l = 0; l < k; l++){
                        real_t dist = distance(centroids + D*l, Y + D*v);
                        if (loss != quadratic_loss()){
                            dist -= bottom_dist[l];
                        }
                        if (dist < nearest_dist[i]){ nearest_dist[i] = dist; }
                    }
                    if (vert_weights){ nearest_dist[i] *= vert_weights[v]; }
                }
                discrete_distribution<index_t> dist_distr(nearest_dist,
                    nearest_dist + comp_size);
                rand_i = dist_distr(rand_gen);
            }
            index_t rand_v = comp_list_rv[rand_i];
            const real_t* Yv = Y + D*rand_v;
            real_t* Ck = centroids + D*k;
            for (size_t d = 0; d < D; d++){ Ck[d] = Yv[d]; }
            if (loss != quadratic_loss()){ bottom_dist[k] = distance(Ck, Ck); }
        } // end for k

        /**  k-means  **/
        for (int kmpp_iter = 0; kmpp_iter < kmpp_iter_num; kmpp_iter++){
            /* assign clusters to centroids */
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                real_t min_dist = real_inf();
                for (comp_t k = 0; k < K; k++){
                    real_t dist = distance(centroids + D*k, Y + D*v);
                    if (dist < min_dist){
                        min_dist = dist;
                        label_assign[v] = k;
                    }
                }
            }
            /* update centroids of clusters */
            update_split_values(rv, centroids);
        }

        /**  compare resulting sum of distances and keep the best one  **/
        real_t sum_dist = ZERO;
        for (index_t i = 0; i < comp_size; i++){
            index_t v = comp_list_rv[i];
            comp_t k = label_assign[v];
            sum_dist += VERT_WEIGHTS_(v)*distance(centroids + D*k, Y + D*v);
        }
        if (sum_dist < min_sum_dist){
            min_sum_dist = sum_dist;
            for (size_t dk = 0; dk < D*K; dk++){ altX[dk] = centroids[dk]; }
            for (index_t i = 0; i < comp_size; i++){
                index_t v = comp_list_rv[i];
                best_assign[i] = label_assign[v];
            }
        }

    } // end for kmpp_init

    free(bottom_dist);
    free(centroids);
    free(nearest_dist);

    /**  copy best label assignment  **/
    for (index_t i = 0; i < comp_size; i++){
        index_t v = comp_list_rv[i];
        label_assign[v] = best_assign[i];
    }

    free(best_assign);
}

TPL void CP_D0_DIST::update_split_values(comp_t rv, real_t* altX)
{
    real_t* total_weights = (real_t*) malloc_check(sizeof(real_t)*K);
    for (comp_t k = 0; k < K; k++){
        total_weights[k] = ZERO;
        real_t* altXk = altX + D*k;
        for (size_t d = 0; d < D; d++){ altXk[d] = ZERO; }
    }
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        index_t v = comp_list[i];
        comp_t k = label_assign[v];
        total_weights[k] += VERT_WEIGHTS_(v);
        const real_t* Yv = Y + D*v;
        real_t* altXk = altX + D*k;
        for (size_t d = 0; d < D; d++){ altXk[d] += VERT_WEIGHTS_(v)*Yv[d]; }
    }
    for (comp_t k = 0; k < K; k++){
        real_t* altXk = altX + D*k;
        if (total_weights[k]){
            for (size_t d = 0; d < D; d++){ altXk[d] /= total_weights[k]; }
        }else{ // no vertex assigned to k, flag with infinity
            altXk[0] = real_inf();
        }
    }
    free(total_weights);
}

TPL bool CP_D0_DIST::is_split_value(real_t altX){ return altX != real_inf(); }

TPL void CP_D0_DIST::update_merge_candidate(index_t re, comp_t ru, comp_t rv)
{
    real_t* rXu = rX + D*ru;
    real_t* rXv = rX + D*rv;
    real_t wru = comp_weights[ru];
    real_t wrv = comp_weights[rv];

    if (loss == quadratic_loss()){
        real_t gain = reduced_edge_weights[re]
            - wru*wrv/(wru + wrv)*distance(rXu, rXv);

        if (gain > ZERO || wru < min_comp_weight || wrv < min_comp_weight){
            if (!merge_info_list[re]){
                merge_info_list[re] = new Merge_info(D);
            }
            merge_info_list[re]->gain = gain;
            wru /= (comp_weights[ru] + comp_weights[rv]);
            wrv /= (comp_weights[ru] + comp_weights[rv]);
            for (size_t d = 0; d < D; d++){
                merge_info_list[re]->value[d] = wru*rXu[d] + wrv*rXv[d];
            }
        }else if (merge_info_list[re]){
            delete merge_info_list[re];
            merge_info_list[re] = nullptr;
        }
    }else{
        if (!merge_info_list[re]){ merge_info_list[re] = new Merge_info(D); }
        real_t* value = merge_info_list[re]->value;

        wru /= (comp_weights[ru] + comp_weights[rv]);
        wrv /= (comp_weights[ru] + comp_weights[rv]);
        for (size_t d = 0; d < D; d++){ value[d] = wru*rXu[d] + wrv*rXv[d]; }

        /* in the following some computations might be saved by factoring
         * multiplications and logarithms, at the cost of readability */
        merge_info_list[re]->gain = reduced_edge_weights[re]
            + comp_weights[ru]*(distance(rXu, rXu) - distance(rXu, value))
            + comp_weights[rv]*(distance(rXv, rXv) - distance(rXv, value));

        if (merge_info_list[re]->gain <= ZERO &&
            comp_weights[ru] >= min_comp_weight &&
            comp_weights[rv] >= min_comp_weight){
            delete merge_info_list[re];
            merge_info_list[re] = nullptr;
        }
    }
}

TPL size_t CP_D0_DIST::update_merge_complexity()
{ return rE*2*D; /* each update is only linear in D */ }

TPL void CP_D0_DIST::accept_merge_candidate(index_t re, comp_t& ru, comp_t& rv)
{
    Cp_d0<real_t, index_t, comp_t>::accept_merge_candidate(re, ru, rv);
        // ru now the root of the merge chain
    comp_weights[ru] += comp_weights[rv];
}

TPL index_t CP_D0_DIST::merge()
{
    index_t deactivation = Cp_d0<real_t, index_t, comp_t>::merge();
    free(comp_weights); comp_weights = nullptr;
    return deactivation;
}

TPL real_t CP_D0_DIST::compute_evolution(bool compute_dif)
{
    if (!compute_dif){ return real_inf(); }
    real_t dif = ZERO;
    #pragma omp parallel for schedule(dynamic) reduction(+:dif) \
        NUM_THREADS(D*(V - saturated_vert), rV)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated[rv]){ continue; }
        real_t* rXv = rX + D*rv;
        real_t distXX = loss == quadratic_loss() ? ZERO : distance(rXv, rXv);
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            real_t* lrXv = last_rX + D*last_comp_assign[v];
            dif += VERT_WEIGHTS_(v)*(distance(rXv, lrXv) - distXX);
        }
    }
    real_t amp = compute_f();
    return amp > eps ? dif/amp : dif/eps;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_d0_dist<float, int32_t, int16_t>;
template class Cp_d0_dist<double, int32_t, int16_t>;
template class Cp_d0_dist<float, int32_t, int32_t>;
template class Cp_d0_dist<double, int32_t, int32_t>;
#else
template class Cp_d0_dist<float, uint32_t, uint16_t>;
template class Cp_d0_dist<double, uint32_t, uint16_t>;
template class Cp_d0_dist<float, uint32_t, uint32_t>;
template class Cp_d0_dist<double, uint32_t, uint32_t>;
#endif
