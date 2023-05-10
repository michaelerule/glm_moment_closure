# Autoregressive Point Processes as Latent State-Space Models: A Moment-Closure Approach to Fluctuations and Autocorrelations

Rule, M., & Sanguinetti, G. (2018). Autoregressive point processes as latent State-Space models: A Moment-Closure approach to fluctuations and autocorrelations. Neural computation, 30(10), 2757-2780. [doi: https://doi.org/10.1162/neco_a_01121.](https://doi.org/10.1162/neco_a_01121)

### Abstract

Modeling and interpreting spike train data is a task of central importance in computational neuroscience, with significant translational implications. Two popular classes of data-driven models for this task are autoregressive point-process generalized linear models (PPGLM) and latent state-space models (SSM) with point-process observations. In this letter, we derive a mathematical connection between these two classes of models. By introducing an auxiliary history process, we represent exactly a PPGLM in terms of a latent, infinite-dimensional dynamical system, which can then be mapped onto an SSM by basis function projections and moment closure. This representation provides a new perspective on widely used methods for modeling spike data and also suggests novel algorithmic approaches to fitting such models. We illustrate our results on a phasic bursting neuron model, showing that our proposed approach provides an accurate and efficient way to capture neural dynamics.

### Repository Contents

This repository contains a demonstration of moment closure applied to autoregressive point-process generalized linear models (PPGLMs), to accompany the paper ["Autoregressive Point Processes as Latent State-Space Models: A Moment-Closure Approach to Fluctuations and Autocorrelations "](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01121). Please see the iPython notebook `ARPPGLM_moment_equations` for further details. 

 - `ARPPGLM_moment_equations.ipynb`: primary demonstration notebook to accompany the paper.
 - `ARPPGLM_gamma_moment_closure_v1.ipynb` and `v2`: Unpublished experiments with other moment-matching assumptions that are more accurate on the simulated test data, but with unclear generality
 - `arppglm.py`: Functions for Langevin sampling and moment propagation
 - `izh.py`: Succinct module to simulate Izhikevich neurons
 - `glm.py`: Routines for autoregressive log-linear Poisson generalized linear models
 - `plot.py`: Plotting helpers
 - `utilities.py`: Misc. helper functions
 - `functions.py`: Defines commonly used small functions
 - `arguments.py`: routines for argument checking and verification

### Example figures

<div style="text-align: center;">
  <img src="./figures/20180808_example_stimulus.png" width="75%" class="img-responsive" style="margin:0 auto; display:block;">
</div>

> ***Caption:*** *Moment closure captures slow timescales in the mean and fast timescales in the variance. Five approaches for approximating the mean (black trace) and variance (shaded, 1σ ) of the log-intensity of the autoregressive PPGLM phasic bursting model, shown here in response to a 150 ms, 0.3 pA current pulse stimulus (vertical black lines). The Langevin equation retains essential slow-timescale features of point process, but moments must be estimated via computationally intensive Monte-Carlo sampling. The mean-field limit with linear noise approximation cannot capture the effects of fluctuations on the mean. Gaussian moment closure captures the influence of second-order statistics on the evolution of the mean, but underestimates the variance owing to incorrectly modeled skewness. A second-order approximation better captures the second moment. An (experimental) moment-closure based on the gamma distribution provides the most accurate recovery of the mean.*


<div style="text-align: center;">
  <img src="./figures/ARPPGLM.gif" width="75%" class="img-responsive" style="margin:0 auto; display:block;">
</div>

> ***Caption:*** *Moment closure methods provide a way to convert between autoregressive and state-space point-process models. The history of the point process is taken as the state space, and moment-closure provides (nonlinear) equations governing the time evolution of the latent distribution over this state-space, approximated as a Gaussian process (red).*




