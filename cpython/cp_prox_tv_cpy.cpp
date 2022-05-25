/*=============================================================================
 * Comp, rX, [List, Gtv, Obj, Time, Dif] = cp_prox_tv_cpy(Y, first_edge,
 *          adj_vertices, edge_weights, cp_dif_tol, cp_it_max, pfdr_rho,
 *          pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
 *          max_num_threads, balance_parallel_split, real_is_double,
 *          compute_List, compute_Subgrads, compute_Obj, compute_Time,
 *          compute_Dif)
 * 
 *  Baudoin Camille 2019, Raguet Hugo 2021
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cp_prox_tv.hpp" 

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t index_t;
    #define NPY_IND NPY_INT32
    /* comment the following if more than 32767 components are expected */
    typedef int16_t comp_t;
    #define NPY_COMP NPY_INT16
    /* uncomment the following if more than 32767 components are expected */
    // typedef int32_t comp_t;
    // #define NPY_COMP NPY_INT32
#else
    typedef uint32_t index_t;
    #define NPY_IND NPY_UINT32
    /* comment the following if more than 65535 components are expected */
    typedef uint16_t comp_t;
    #define NPY_COMP NPY_UINT16
    /* uncomment the following if more than 65535 components are expected */
    // typedef uint32_t comp_t;
    // #define NPY_COMP NPY_UINT32
#endif

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES NPY_REAL>
static PyObject* cp_prox_tv(PyArrayObject* py_Y, PyArrayObject* py_first_edge,
    PyArrayObject* py_adj_vertices, PyArrayObject* py_edge_weights,
    real_t cp_dif_tol, int cp_it_max, real_t pfdr_rho, real_t pfdr_cond_min,
    real_t pfdr_dif_rcd, real_t pfdr_dif_tol, int pfdr_it_max, int verbose,
    int max_num_threads, int balance_parallel_split, int compute_List,
    int compute_Subgrads, int compute_Obj, int compute_Time, int compute_Dif)
{
    /**  get inputs  **/

    /* square l2 */
    const real_t* Y = (real_t*) PyArray_DATA(py_Y);
    index_t V = PyArray_SIZE(py_Y);

    /* graph structure */
    index_t E = PyArray_SIZE(py_adj_vertices);
    const index_t* first_edge = (index_t*) PyArray_DATA(py_first_edge); 
    const index_t* adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices); 

    /* penalizations */
    const real_t* edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = PyArray_SIZE(py_edge_weights) == 1 ?
        edge_weights[0] : 1.0;
    if (PyArray_SIZE(py_edge_weights) <= 1){ edge_weights = nullptr; }

    /* number of threads */ 
    if (max_num_threads <= 0){ max_num_threads = omp_get_max_threads(); }

    /**  prepare output; rX is created later  **/
    /* NOTA: no check for successful allocations is performed */

    npy_intp size_py_Comp[] = {V};
    PyArrayObject* py_Comp = (PyArrayObject*) PyArray_Zeros(1, size_py_Comp,
        PyArray_DescrFromType(NPY_COMP), 1);
    comp_t* Comp = (comp_t*) PyArray_DATA(py_Comp); 

    real_t* Gtv = nullptr;
    PyArrayObject* py_Gtv = nullptr;;
    if (compute_Subgrads){
        npy_intp size_py_Gtv[] = {E};
        py_Gtv = (PyArrayObject*) PyArray_Zeros(1, size_py_Gtv,
            PyArray_DescrFromType(NPY_REAL), 1);
        Gtv = (real_t*) PyArray_DATA(py_Gtv);
    }

    real_t* Obj = nullptr;
    if (compute_Obj){ Obj = (real_t*) malloc(sizeof(real_t)*(cp_it_max + 1)); }

    double* Time = nullptr;
    if (compute_Time){
        Time = (double*) malloc(sizeof(double)*(cp_it_max + 1));
    }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) malloc(sizeof(real_t)*cp_it_max); }

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_prox_tv<real_t, index_t, comp_t>* cp =
        new Cp_prox_tv<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices);

    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_observation(Y);
    cp->set_d1_subgradients(Gtv);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp); // use the preallocated component array Comp

    int cp_it = cp->cut_pursuit();

    /* retrieve monitoring arrays */
    PyArrayObject* py_Obj = nullptr;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {cp_it + 1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Obj_ = (real_t*) PyArray_DATA(py_Obj);
        for (int i = 0; i < size_py_Obj[0]; i++){ Obj_[i] = Obj[i]; }
        free(Obj);
    }

    PyArrayObject* py_Time = nullptr;
    if (compute_Time){
        npy_intp size_py_Time[] = {cp_it + 1};
        py_Time = (PyArrayObject*) PyArray_Zeros(1, size_py_Time,
            PyArray_DescrFromType(NPY_FLOAT64), 1);
        double* Time_ = (double*) PyArray_DATA(py_Time);
        for (int i = 0; i <= size_py_Time[0]; i++){ Time_[i] = Time[i]; }
        free(Time);
    }

    PyArrayObject* py_Dif = nullptr;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {cp_it};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Dif_ = (real_t*) PyArray_DATA(py_Dif);
        for (int i = 0; i < size_py_Dif[0]; i++){ Dif_[i] = Dif[i]; }
        free(Dif);
    }

    /* get number of components and their lists of indices if necessary */
    index_t *first_vertex, *comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);

    PyObject* py_List = nullptr;
    if (compute_List){
        py_List = PyList_New(rV); // list of arrays
        for (comp_t rv = 0; rv < rV; rv++){
            index_t comp_size = first_vertex[rv+1] - first_vertex[rv];
            npy_intp size_py_List_rv[] = {comp_size};
            PyArrayObject* py_List_rv = (PyArrayObject*) PyArray_Zeros(1,
                size_py_List_rv, PyArray_DescrFromType(NPY_IND), 1);
            index_t* List_rv = (index_t*) PyArray_DATA(py_List_rv);
            for (index_t i = 0; i < comp_size; i++){
                List_rv[i] = comp_list[first_vertex[rv] + i];
            }
            PyList_SetItem(py_List, rv, (PyObject*) py_List_rv);
        }
    }

    /* copy reduced values */
    real_t* cp_rX = cp->get_reduced_values();
    npy_intp size_py_rX[] = {rV};
    PyArrayObject* py_rX = (PyArrayObject*) PyArray_Zeros(1, size_py_rX,
        PyArray_DescrFromType(NPY_REAL), 1);
    real_t* rX = (real_t*) PyArray_DATA(py_rX);
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }

    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /* build output according to optional output specified */
    if (compute_List && compute_Subgrads && compute_Obj && compute_Time &&
        compute_Dif){
        return Py_BuildValue("OOOOOOO", py_Comp, py_rX, py_List, py_Gtv,
            py_Obj, py_Time, py_Dif);
    }else if (compute_List && compute_Subgrads && compute_Obj && compute_Time){
        return Py_BuildValue("OOOOOO", py_Comp, py_rX, py_List, py_Gtv,
            py_Obj, py_Time);
    }else if (compute_List && compute_Subgrads && compute_Obj && compute_Dif){
        return Py_BuildValue("OOOOOO", py_Comp, py_rX, py_List, py_Gtv,
            py_Obj, py_Dif);
    }else if (compute_List && compute_Subgrads && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOOO", py_Comp, py_rX, py_List, py_Gtv,
            py_Time, py_Dif);
    }else if (compute_List && compute_Obj && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOOO", py_Comp, py_rX, py_List, py_Obj,
            py_Time, py_Dif);
    }else if (compute_Subgrads && compute_Obj && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOOO", py_Comp, py_rX, py_Gtv, py_Obj,
            py_Time, py_Dif);
    }else if (compute_List && compute_Subgrads && compute_Obj){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Gtv, py_Obj);
    }else if (compute_List && compute_Subgrads && compute_Time){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Gtv,
            py_Time);
    }else if (compute_List && compute_Subgrads && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Gtv, py_Dif);
    }else if (compute_List && compute_Obj && compute_Time){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Obj,
            py_Time);
    }else if (compute_List && compute_Obj && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Obj, py_Dif);
    }else if (compute_List && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_List, py_Time,
            py_Dif);
    }else if (compute_Subgrads && compute_Obj && compute_Time){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_Gtv, py_Obj, py_Time);
    }else if (compute_Subgrads && compute_Obj && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_Gtv, py_Obj, py_Dif);
    }else if (compute_Subgrads && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_Gtv, py_Time, py_Dif);
    }else if (compute_Obj && compute_Time && compute_Dif){
        return Py_BuildValue("OOOOO", py_Comp, py_rX, py_Obj, py_Time, py_Dif);
    }else if (compute_List && compute_Subgrads){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_List, py_Gtv);
    }else if (compute_List && compute_Obj){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_List, py_Obj);
    }else if (compute_List && compute_Time){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_List, py_Time);
    }else if (compute_List && compute_Dif){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_List, py_Dif);
    }else if (compute_Subgrads && compute_Obj){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Gtv, py_Obj);
    }else if (compute_Subgrads && compute_Time){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Gtv, py_Time);
    }else if (compute_Subgrads && compute_Dif){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Gtv, py_Dif);
    }else if (compute_Obj && compute_Time){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Obj, py_Time);
    }else if (compute_Obj && compute_Dif){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Obj, py_Dif);
    }else if (compute_Time && compute_Dif){
        return Py_BuildValue("OOOO", py_Comp, py_rX, py_Time, py_Dif);
    }else if (compute_List){
        return Py_BuildValue("OOO", py_Comp, py_rX, py_List);
    }else if (compute_Subgrads){
        return Py_BuildValue("OOO", py_Comp, py_rX, py_Gtv);
    }else if (compute_Obj){
        return Py_BuildValue("OOO", py_Comp, py_rX, py_Obj);
    }else if (compute_Time){
        return Py_BuildValue("OOO", py_Comp, py_rX, py_Time);
    }else if (compute_Dif){
        return Py_BuildValue("OOO", py_Comp, py_rX, py_Dif);
    }else{
        return Py_BuildValue("OO", py_Comp, py_rX);
    }

}

/* actual interface */
#if PY_VERSION_HEX >= 0x03040000 // Py_UNUSED suppress warning from 3.4
static PyObject* cp_prox_tv_cpy(PyObject* Py_UNUSED(self), PyObject* args)
{ 
#else
static PyObject* cp_prox_tv_cpy(PyObject* self, PyObject* args)
{   (void) self; // suppress unused parameter warning
#endif
    /* INPUT */ 
    PyArrayObject *py_Y, *py_first_edge, *py_adj_vertices, *py_edge_weights; 
    double cp_dif_tol, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int cp_it_max, pfdr_it_max, verbose, max_num_threads, 
        balance_parallel_split, real_is_double, compute_List, compute_Subgrads,
        compute_Obj, compute_Time, compute_Dif; 
    
    /* parse the input, from Python Object to C PyArray, double, or int type */
    if(!PyArg_ParseTuple(args, "OOOOdiddddiiiiiiiiii", &py_Y, &py_first_edge,
        &py_adj_vertices, &py_edge_weights, &cp_dif_tol, &cp_it_max, &pfdr_rho,
        &pfdr_cond_min, &pfdr_dif_rcd, &pfdr_dif_tol, &pfdr_it_max, &verbose,
        &max_num_threads, &balance_parallel_split, &real_is_double,
        &compute_List, &compute_Subgrads, &compute_Obj, &compute_Time,
        &compute_Dif)){
        return NULL;
    }

    if (real_is_double){ /* real_t type is double */
        return cp_prox_tv<double, NPY_FLOAT64>(py_Y, py_first_edge,
            py_adj_vertices, py_edge_weights, cp_dif_tol, cp_it_max, pfdr_rho,
            pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
            max_num_threads, balance_parallel_split, compute_List,
            compute_Subgrads, compute_Obj, compute_Time, compute_Dif);
    }else{ /* real_t type is float */
        return cp_prox_tv<float, NPY_FLOAT32>(py_Y, py_first_edge,
            py_adj_vertices, py_edge_weights, cp_dif_tol, cp_it_max, pfdr_rho,
            pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
            max_num_threads, balance_parallel_split, compute_List,
            compute_Subgrads, compute_Obj, compute_Time, compute_Dif);
    }
}

static PyMethodDef cp_prox_tv_methods[] = {
    {"cp_prox_tv_cpy", cp_prox_tv_cpy, METH_VARARGS,
        "wrapper for parallel cut-pursuit prox TV"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef cp_prox_tv_module = {
    PyModuleDef_HEAD_INIT,
    "cp_prox_tv_cpy", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    cp_prox_tv_methods,
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_cp_prox_tv_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&cp_prox_tv_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initcp_prox_tv_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    (void) Py_InitModule("cp_prox_tv_cpy", cp_prox_tv_methods);
}

#endif
