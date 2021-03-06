/*=============================================================================
 * Hugo Raguet 2016
 *===========================================================================*/
#include "omp_num_threads.hpp"
#include "pcd_fwd_doug_rach.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.)
#define ONE ((real_t) 1.)
#define TWO ((real_t) 2.)
#define ALMOST_TWO ((real_t) 1.9)

/* macros for indexing conditioners depending on their shape */
#define W_(j, jd)    (wshape == SCALAR ? W[(j)/size] : \
                      wshape == MONODIM ? W[(j)] : W[(jd)])
#define Ga_(i, id)   (gashape == SCALAR ? ga : \
                      gashape == MONODIM ? Ga[(i)] : Ga[(id)])
#define L_(i, id)    (lshape == SCALAR ? l : \
                      lshape == MONODIM ? L[(i)] : L[(id)])
#define Id_W_(i, id) (wshape == MONODIM ? Id_W[(i)] : Id_W[(id)])
#define aux_idx_(j) (aux_idx ? aux_idx[(j)] : ((j) % size))

#define TPL template <typename real_t, typename var_index_t>
#define PFDR Pfdr<real_t, var_index_t>

using namespace std;

TPL PFDR::Pfdr(var_index_t size, index_t aux_size, const var_index_t* aux_idx,
    index_t D, Condshape gashape, Condshape wshape) : Pcd_prox<real_t>(size*D),
        size(size), aux_size(aux_size), D(D), aux_idx(aux_idx),
        gashape(gashape), wshape(wshape)
{
    set_name("Preconditioned forward-Douglas-Rachford algorithm");
    rho = 1.5;
    L = Lmut = nullptr;
    l = ZERO; lshape = SCALAR;
    lipschcomput = EACH;
    Ga = Ga_grad_f = Z = W = Z_Id = Id_W = nullptr;
}

TPL PFDR::~Pfdr(){ free(Ga); free(Z); free(W); free(Ga_grad_f); free(Lmut); }

TPL void PFDR::set_relaxation(real_t rho){ this->rho = rho; }

TPL void PFDR::set_lipschitz_param(const real_t* L, real_t l, Condshape lshape)
{
    this->L = L;
    this->l = l;
    if (L){ this->lshape = lshape < gashape ? lshape : gashape; }
    else{ this->lshape = SCALAR; }
    this->lipschcomput = USER;
}

TPL void PFDR::set_lipschitz_param(Lipschcomput lipschcomput)
{ this->lipschcomput = lipschcomput; }

TPL void PFDR::set_auxiliary(real_t* Z){ this->Z = Z; }

TPL real_t* PFDR::get_auxiliary(){ return this->Z; }

TPL void PFDR::compute_lipschitz_metric(){ l = ZERO; lshape = SCALAR; }

TPL void PFDR::compute_hess_f()
/* default to zero f, can be overriden */
{
    for (var_index_t i = 0; i < size; i++){
        index_t id = i*D;
        for (index_t d = 0; d < D; d++){ Ga_(i, id++) = ZERO; }
    }
}

TPL void PFDR::add_pseudo_hess_h() /* default to zero h, can be overriden */ {}

TPL void PFDR::make_sum_Wi_Id()
{
    if (wshape == SCALAR){
        index_t n = aux_size/size;
        real_t sum_wi = ZERO;
        for (var_index_t i = 0; i < n; i++){ sum_wi += W[i]; }
        for (var_index_t i = 0; i < n; i++){ W[i] /= sum_wi; }
    }else{
        if (!Id_W){ /* normalize by the sum */
            const index_t Dw = wshape == MULTIDIM ? D : 1;
            /* compute sum */
            real_t* sum_Wi = (real_t*) malloc_check(sizeof(real_t)*size*Dw);
            for (index_t id = 0; id < size*Dw; id++){ sum_Wi[id] = ZERO; }
            #pragma omp parallel for schedule(static) \
                NUM_THREADS(aux_size*Dw, Dw)
            for (index_t d = 0; d < Dw; d++){
                index_t jd = d;
                for (index_t j = 0; j < aux_size; j++){
                    sum_Wi[d + aux_idx_(j)*Dw] += W[jd];
                    jd += Dw;
                }
            }
            /* normalize */
            #pragma omp parallel for schedule(static) \
                NUM_THREADS(aux_size*Dw, aux_size)
            for (index_t j = 0; j < aux_size; j++){
                index_t id = aux_idx_(j)*Dw;
                index_t jd = j*Dw;
                for (index_t d = 0; d < D; d++){ W[jd++] /= sum_Wi[id++]; }
            }
            free(sum_Wi);
        }else{ /* metric must be kept */            
            cerr << "PFDR: a specialization of the virtual "
                "function make_sum_Wi_Id() must be provided in order to use "
                "the weights Wi to shape the metric of the proximity operators"
                " of g." << endl;
            exit(EXIT_FAILURE);
        }
    }
}

TPL void PFDR::initialize_auxiliary(){
    if (!Z){ Z = (real_t*) malloc_check(sizeof(real_t)*aux_size*D); }
    for (index_t j = 0; j < aux_size; j++){
        index_t id = aux_idx_(j)*D;
        index_t jd = j*D;
        for (index_t d = 0; d < D; d++){ Z[jd++] = X[id++]; }
    }
    if (Z_Id){ for (index_t id = 0; id < size*D; id++){ Z_Id[id] = X[id]; } }
}

TPL void PFDR::compute_Ga_grad_f()
/* default to zero f, can be overriden */
{ for (index_t id = 0; id < size*D; id++){ Ga_grad_f[id] = ZERO; } }

TPL void PFDR::compute_weighted_average()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(aux_size*D, D)
    for (index_t d = 0; d < D; d++){ 
        index_t id = d;
        for (var_index_t i = 0; i < size; i++){
            X[id] = !Id_W ? ZERO : 
                (Id_W_(i, id)*(Z_Id ? Z_Id[id] : Ga_grad_f[id] - X[id]));
            id += D;
        }
        index_t jd = d;
        for (index_t j = 0; j < aux_size; j++){
            X[d+aux_idx_(j)*D] += W_(j, jd)*Z[jd];
            jd += D;
        }
    }
}

TPL void PFDR::compute_prox_Ga_h()
/* default to zero f, can be overriden */ {}

TPL real_t PFDR::compute_f()
/* default to zero f, can be overriden */ { return ZERO; }

TPL real_t PFDR::compute_h()
/* default to zero f, can be overriden */ { return ZERO; }

TPL void PFDR::preconditioning(bool init)
{
    Pcd_prox<real_t>::preconditioning(init);

    if (init){
        if (!Z){ initialize_auxiliary(); }
        if (!Ga && gashape != SCALAR){
            if (gashape == MONODIM){
                Ga = (real_t*) malloc_check(sizeof(real_t)*size);
            }else{
                Ga = (real_t*) malloc_check(sizeof(real_t)*size*D);
            }
        }
        if (!W){
            if (wshape == SCALAR){
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size/size);
            }else if (wshape == MONODIM){
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size);
            }else{
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size*D);
            }
        }
        if (!Ga_grad_f){
            Ga_grad_f = (real_t*) malloc_check(sizeof(real_t)*size*D);
        }
    }else{ /**  compute the auxiliary subgradients in Z  **/
        compute_Ga_grad_f(); 
        #pragma omp parallel for schedule(static) \
            NUM_THREADS(D*4*aux_size, aux_size)
        for (index_t j = 0; j < aux_size; j++){
            var_index_t i = aux_idx_(j);
            index_t id = i*D;
            index_t jd = j*D;
            for (index_t d = 0; d < D; d++){
                Z[jd] = (W_(j, jd)/Ga_(i, id))*(X[id] - Ga_grad_f[id] - Z[jd]);
                id++; jd++;
            }
        }
        if (Z_Id){
            #pragma omp parallel for schedule(static) NUM_THREADS(D*size, size)
            for (var_index_t i = 0; i < size; i++){
                index_t id = i*D;
                for (index_t d = 0; d < D; d++){
                    Z_Id[id] = (Id_W_(i, id)/Ga_(i, id))
                        *(X[id] - Ga_grad_f[id] - Z_Id[id]);
                    id++;
                }
            }
        }
    }

    /**  second-order information on f  **/
    if (lipschcomput == EACH || (lipschcomput == ONCE && !L && !l)){
        compute_lipschitz_metric();
    }

    compute_hess_f();

    add_pseudo_hess_g();

    add_pseudo_hess_h();

    const index_t Dga = gashape == MULTIDIM ? D : 1;
    const var_index_t sizega = gashape == SCALAR ? 1 : size;

    /**  inverse the pseudo-Hessian  **/
    #pragma omp parallel for schedule(static) NUM_THREADS(sizega*Dga)
    for (index_t id = 0; id < sizega*Dga; id++){
        Ga_(id, id) = ONE/Ga_(id, id);
    }

    /**  convergence condition on the metric and stability  **/ 
    real_t lga_max = ALMOST_TWO*(TWO - rho);
    #pragma omp parallel for schedule(static) NUM_THREADS(sizega*Dga, sizega)
    for (var_index_t i = 0; i < sizega; i++){
        index_t id = i*Dga;
        for (index_t d = 0; d < Dga; d++){
            real_t ga_max = lga_max/L_(i, id);
            real_t ga_min = L_(i, id) > ZERO ? cond_min*ga_max : cond_min;
            if (Ga_(i, id) > ga_max){ Ga_(i, id) = ga_max; }
            else if (Ga_(i, id) < ga_min){ Ga_(i, id) = ga_min; }
            id++;
        }
    }

    if (lipschcomput == EACH){ free(Lmut); L = Lmut = nullptr; }

    make_sum_Wi_Id();

    if (!init){ /**  update auxiliary variables  **/
        #pragma omp parallel for schedule(static) \
            NUM_THREADS(2*aux_size*D, aux_size)
        for (index_t j = 0; j < aux_size; j++){
            var_index_t i = aux_idx_(j);
            index_t id = i*D;
            index_t jd = j*D;
            for (index_t d = 0; d < D; d++){
                Z[jd] = X[id] - Ga_grad_f[id] - Ga_(i, id)*Z[jd]/W_(j, jd);
                id++; jd++;
            }
        }
        if (Z_Id){
            #pragma omp parallel for schedule(static) NUM_THREADS(D*size, size)
            for (var_index_t i = 0; i < size; i++){
                index_t id = i*D;
                for (index_t d = 0; d < D; d++){
                    Z_Id[id] = X[id] - Ga_grad_f[id] -
                        Ga_(i, id)*Z_Id[id]/Id_W_(i, id);
                    id++;
                }
            }
        }
    }
}

TPL void PFDR::main_iteration()
{
    /* gradient step, forward = 2 X - Zi - Ga grad(X) */ 
    compute_Ga_grad_f();
    #pragma omp parallel for schedule(static) NUM_THREADS(size*D)
    for (index_t id = 0; id < size*D; id++){
        Ga_grad_f[id] = TWO*X[id] - Ga_grad_f[id];
    }

    /* generalized forward-backward step on auxiliary Z */
    compute_prox_GaW_g(); 
    if (Z_Id){ /* take care of the additional auxiliary variable */
        for (index_t id = 0; id < size*D; id++){
            Z_Id[id] += rho*(Ga_grad_f[id] - Z_Id[id] - X[id]); id++;
        }
    }

    /* projection on first diagonal */
    compute_weighted_average();

    /* backward step on iterate X */
    compute_prox_Ga_h(); 
}

TPL real_t PFDR::compute_objective()
{ return compute_f() + compute_g() + compute_h(); }

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr<float, int16_t>;
template class Pfdr<float, int32_t>;
template class Pfdr<double, int16_t>;
template class Pfdr<double, int32_t>;
#else
template class Pfdr<float, uint16_t>;
template class Pfdr<float, uint32_t>;
template class Pfdr<double, uint16_t>;
template class Pfdr<double, uint32_t>;
#endif
