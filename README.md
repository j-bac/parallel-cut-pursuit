# Cut-Pursuit Algorithms, Parallelized Along Components
Fork of official repository from Hugo Raguet (https://gitlab.com/1a7r0ch3/parallel-cut-pursuit) that contains only the extension modules for Python.  

Generic C++ classes for implementing cut-pursuit algorithms.  
Specialization to convex problems involving **graph total variation**, and nonconvex problems involving **contour length**, as explained in our articles [(Landrieu and Obozinski, 2016; Raguet and Landrieu, 2018)](#references).   


## Table of Content  

1. [**General problem statement**](#general-problem-statement)  
2. [**C++ classes and Specializations**](#c-classes-pecializations)  
2.1. [Proximity operator of the graph total variation](#cp_prox_tv-proximity-operator-of-the-graph-total-variation)  
2.2. [Quadratic functional and graph total variation](#cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation)  
2.3. [Separable multidimensional loss and graph total variation](#cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation)  
2.4. [Separable distance and contour length](#cp_d0_dist-separable-distance-and-weighted-contour-length)  
3. [**Documentation**](#documentation)  
3.1. [Directory tree](#directory-tree)  
3.2. [Graph structure](#graph-structure)  
3.3. [Python](#python)  
4. [**References**](#references)  
5. [**License**](#license)  

## General problem statement
The cut-pursuit algorithms minimize functionals structured, over a weighted graph _G_ = (_V_, _E_, _w_), as 

    _F_: _x_ ∈ Ω<sup>_V_</sup> ↦ _f_(_x_) + 
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>  _ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) ,    

where Ω is some base set, and the functional _ψ_: Ω² → ℝ penalizes dissimilarity between its arguments, in order to enforce solutions which are *piecewise constant along the graph _G_*.

The cut-pursuit approach is to seek partitions __*V*__ of the set of vertices _V_, constituting the constant connected components of the solution, by successively solving the corresponding problem, structured over the reduced graph __*G*__ = (__*V*__, __*E*__), that is

  arg min<sub>_ξ_ ∈ Ω<sup>__*V*__</sup></sub>
  _F_(_x_) ,    such that ∀ _U_ ∈ __*V*__, ∀ _u_ ∈ _U_, _x_<sub>_u_</sub> = _ξ_<sub>_U_</sub> ,

and then refining the partition.  
A key requirement is thus the ability to solve the reduced problem, which often have the exact same structure as the original one, but with much less vertices |__*V*__| ≪ |_V_|. If the solution of the original problem has only few constant connected components in comparison to the number of vertices, the cut-pursuit strategy can speed-up minimization by several orders of magnitude.  

Cut-pursuit algorithms come in two main flavors, namely “directionally differentiable” and “noncontinuous”.

* In the **directionally differentiable** case, the base set Ω is typically a vector space, and it is required that _f_ is differentiable, or at least that its nondifferentiable part is _separable along the graph_ and admits (potentially infinite) _directional derivatives_. This comprises notably many convex problems, where 
_ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) = ║<i>x</i><sub>_u_</sub> − _x_<sub>_v_</sub>║, that is to say involving a _**graph total variation**_. The refinement of the partition is based on the search for a steep directional derivative, and the reduced problem is solved using convex or continuous optimization; optimality guarantees can be provided.  

* In the **noncontinuous** case, the dissimilarity penalization typically uses _ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) = 0 if _x_<sub>_u_</sub> =_x_<sub>_v_</sub>, 1 otherwise, resulting in a measure of the _**contour length**_ of the constant connected components. The functional _f_ is typically required to be separable along the graph, and to have computational properties favorable enough for solving reduced problems. The refinement of the partition relies on greedy heuristics.

Both flavors admit multidimensional extensions, that is to say Ω is not required to be only a set of scalars.

## C++ classes and Specializations

The class `Cp_graph` is a modification of the `Graph` class of Y. Boykov and V. Kolmogorov, for making use of their [maximum flow algorithm](#references).  
The class `Cp` is the most generic, defining all steps of the cut-pursuit approach in virtual methods.  
The class `Cp_d1` specializes methods for directionally differentiable cases involving the graph total variation.  
The class `Cp_d0` specializes methods for noncontinuous cases involving the contour length penalization.  

### `Cp_prox_tv`: proximity operator of the graph total variation
Also coined “graph total variation denoising” or “general fused LASSO signal approximation”. The base set is Ω = ℝ, and the objective functional is  

    _F_: _x_ ∈ ℝ<sup>_V_</sup> ↦  1/2 ║<i>y</i> − <i>x</i>║<sup>2</sup> +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,   

where _y_ ∈ ℝ<sup>_n_</sup> 
and _w_ ∈ ℝ<sup>_E_</sup> are regularization weights.  

### `Cp_d1_ql1b`: quadratic functional, ℓ<sub>1</sub> norm, bounds, and graph total variation
The base set is Ω = ℝ, and the general form is  

    _F_: _x_ ∈ ℝ<sup>_V_</sup> ↦  1/2 ║<i>y</i><sup>(q)</sup> − _A_<i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _λ_<sub>_v_</sub> |_y_<sup>(ℓ<sub>1</sub>)</sup> − _x_<sub>_v_</sub>| +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>[_m_<sub>_v_</sub>, _M_<sub>_v_</sub>]</sub>(_x_<sub>_v_</sub>)   
                 +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,   

where _y_<sup>(q)</sup> ∈ ℝ<sup>_n_</sup>, 
_A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, 
_y_<sup>(ℓ<sub>1</sub>)</sup> ∈ ℝ<sup>_V_</sup> and 
_λ_ ∈ ℝ<sup>_V_</sup> and _w_ ∈ ℝ<sup>_E_</sup> are regularization weights, 
_m_, _M_ ∈ ℝ<sup>_V_</sup> are parameters and 
_ι_<sub>[_a_,_b_]</sub> is the convex indicator of [_a_, _b_] : x ↦ 0 if _x_ ∈ [_a_, _b_], +∞ otherwise.  

When _y_<sup>(ℓ<sub>1</sub>)</sup> is zero, the combination of ℓ<sub>1</sub> norm and total variation is sometimes coined _fused LASSO_.  

When _A_ is the identity, _λ_ is zero and there are no box constraints, the problem boils down to [the proximity operator of the graph total variation](#specialization-cp_prox_tv-proximity-operator-of-the-graph-total-variation).  

Currently, _A_ must be provided as a matrix. See the documentation for special cases.  

The reduced problem is solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/) (see also the [corresponding repository](https://gitlab.com/1a7r0ch3/pcd-prox-split)).  

Two examples where _A_ is a full ill-conditioned matrix are provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) and [Python](#python) interfaces: one with positivity and fused LASSO constraints on a task of _brain source identification from electroencephalography_, and another with boundary constraints on a task of _image reconstruction from tomography_.

<table><tr>
<td width="10%"></td>
<td width="20%"> ground truth </td>
<td width="10%"></td>
<td width="20%"> raw retrieved activity </td>
<td width="10%"></td>
<td width="20%"> identified sources </td>
<td width="10%"></td>
</tr><tr>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_ground_truth.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_brain_activity.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_brain_sources.png" width="100%"/></td>
<td width="10%"></td>
</tr></table>

### Specialization `Cp_d1_lsx`: separable loss, simplex constraints, and graph total variation
The base set is Ω = ℝ<sup>_D_</sup>, where _D_ can be seen as a set of labels, and the general form is  

    _F_: _x_ ∈ ℝ<sup>_V_ ⨯ _D_</sup> ↦  _f_(_y_, _x_) +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>Δ<sub>_D_</sub></sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sup>(d<sub>1</sub>)</sup><sub>(_u_,_v_)</sub>
 ∑<sub>_d_ ∈ _D_</sub> _λ_<sub>_d_</sub> |_x_<sub>_u_,_d_</sub> − _x_<sub>_v_,_d_</sub>| ,  

where _y_ ∈ ℝ<sup>_V_ ⨯ _D_</sup>, _f_ is a loss functional (see below), _w_<sup>(d<sub>1</sub>)</sup> ∈ ℝ<sup>_E_</sup> and _λ_ ∈ ℝ<sup>_D_</sup> are regularization weights, and _ι_<sub>Δ<sub>_D_</sub></sub> is the convex indicator of the simplex
Δ<sub>_D_</sub> = {_x_ ∈ ℝ<sup>_D_</sup> | ∑<sub>_d_</sub> _x_<sub>_d_</sub> = 1 and ∀ _d_, _x_<sub>_d_</sub> ≥ 0}: _x_ ↦ 0 if _x_ ∈ Δ<sub>_D_</sub>, +∞ otherwise. 

The following loss functionals are available, where _w_<sup>(_f_)</sup> ∈ ℝ<sup>_V_</sup> are weights on vertices.  
Linear: _f_(_y_, _x_) = − ∑<sub>_v_ ∈ _V_</sub> _w_<sup>(_f_)</sup><sub>_v_</sub> ∑<sub>_d_ ∈ _D_</sub> _x_<sub>_v_,_d_</sub> _y_<sub>_v_,_d_</sub>  
Quadratic: _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> _w_<sup>(_f_)</sup><sub>_v_</sub> ∑<sub>_d_ ∈ _D_</sub> (_x_<sub>_v_,_d_</sub> − _y_<sub>_v_,_d_</sub>)<sup>2</sup>  
Smoothed Kullback–Leibler divergence (equivalent to cross-entropy):  
_f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> _w_<sup>(_f_)</sup><sub>_v_</sub>
KL(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),  
where _α_ ∈ \]0,1\[,
_u_ ∈ Δ<sub>_D_</sub> is the uniform discrete distribution over _D_,
and
KL: (_p_, _q_) ↦ ∑<sub>_d_ ∈ _D_</sub> _p_<sub>_d_</sub> log(_p_<sub>_d_</sub>/_q_<sub>_d_</sub>).  

The reduced problem is also solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/) (see also the [corresponding repository](https://github.com/1a7r0ch3/pcd-prox-split)).  

An example with the smoothed Kullback–Leibler is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) and [Python](#python) interfaces, on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

<table><tr>
<td width="5%"></td>
<td width="25%"> ground truth </td>
<td width="5%"></td>
<td width="25%"> random forest classifier </td>
<td width="5%"></td>
<td width="25%"> regularized classification </td>
<td width="5%"></td>
</tr><tr>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_ground_truth.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_random_forest.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_regularized.png" width="100%"/></td>
<td width="5%"></td>
</tr></table>

### Specialization `Cp_d0_dist`: separable distance and weighted contour length
The base set is Ω = ℝ<sup>_D_</sup> or Δ<sub>_D_</sub> and the general form is  

    _F_: _x_ ∈ ℝ<sup>_V_ ⨯ _D_</sup> ↦  _f_(_y_, _x_) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sup>(d<sub>0</sub>)</sup><sub>(_u_,_v_)</sub>
    ║<i>x</i><sub>_u_</sub> − _x_<sub>_v_</sub>║<sub>0</sub> ,  

where _y_ ∈ Ω<sup>_V_</sup>, _f_ is a loss functional akin to a distance (see below), and 
║&middot;║<sub>0</sub> is the ℓ<sub>0</sub> pseudo-norm _x_ ↦ 0 if _x_ = 0, 1 otherwise.  

The following loss functionals are available, where _w_<sup>(_f_)</sup> ∈ ℝ<sup>_V_</sup> are weights on vertices and _m_<sup>(_f_)</sup> ∈ ℝ<sup>_D_</sup> are weights on coordinates.  
Weighted quadratic: Ω = ℝ<sup>_D_</sup> and 
_f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> _w_<sup>(_f_)</sup><sub>_v_</sub> ∑<sub>_d_ ∈ _D_</sub> _m_<sup>(_f_)</sup><sub>_d_</sub> (_x_<sub>_v_,_d_</sub> − _y_<sub>_v_,_d_</sub>)<sup>2</sup>  
Weighted smoothed Kullback–Leibler divergence (equivalent to cross-entropy):
Ω = Δ<sub>_D_</sub> and  
_f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> _w_<sup>(_f_)</sup><sub>_v_</sub>
KL<sub>_m_<sup>(_f_)</sup></sub>(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),  
where _α_ ∈ \]0,1\[,
_u_ ∈ Δ<sub>_D_</sub> is the uniform discrete distribution over _D_,
and  
KL<sub>_m_<sup>(_f_)</sup></sub>: (_p_, _q_) ↦ ∑<sub>_d_ ∈ _D_</sub> _m_<sup>(_f_)</sup><sub>_d_</sub> _p_<sub>_d_</sub> log(_p_<sub>_d_</sub>/_q_<sub>_d_</sub>).   

The reduced problem amounts to averaging, and the split step uses _k_-means++ algorithm.  

When the loss is quadratic, the resulting problem is sometimes coined “minimal partition problem”.  

An example with the smoothed Kullback–Leibler is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) interface, on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

## Documentation

### Graph structure

Graph structures must be given as forward-star representation. For conversion from simple adjacency list representation, or for creation from scratch for regular N-dimensionnal grids (2D for images, 3D for volumes, etc.), see my [grid graph](https://gitlab.com/1a7r0ch3/grid-graph) repository.  


### Python
Requires `numpy` package.  
See the script `setup.py` for compiling modules with `distutils`; on UNIX systems, it can be directly interpreted as `python setup.py build_ext`.  
Compatible with Python 2 and Python 3.  

Extensive documentation of the Python wrappers can be found in the corresponding `.py` files.  
The scripts are mostly written for Python 3, and should work with Python 2 with minor tweaking.

The script `example_EEG.py` exemplifies the use of [`Cp_d1_ql1b`](#specialization-cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation), on a task of _brain source identification from electroencephalography_.  

The script `example_tomography.py` exemplifies the use of [`Cp_d1_ql1b`](#specialization-cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation), on a task of _image reconstruction from tomography_.   

The scripts `example_labeling_3D.py` and `example_labeling_3D_d0.py` exemplify the use of, respectively, [`Cp_d1_lsx`](#specialization-cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation) and [`Cp_d0_dist`](#specialization-cp_d0_dist-separable-distance-and-weighted-contour-length), on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

## References
L. Landrieu and G. Obozinski, [Cut Pursuit: Fast Algorithms to Learn Piecewise Constant Functions on Weighted Graphs](http://epubs.siam.org/doi/abs/10.1137/17M1113436), 2017.  

H. Raguet and L. Landrieu, [Cut-pursuit Algorithm for Regularizing Nonsmooth Functionals with Graph Total Variation](https://1a7r0ch3.github.io/cp/), 2018.  

Y. Boykov and V. Kolmogorov, An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004.

## License
This software is under the GPLv3 license.
