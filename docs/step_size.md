
# Step Size is a Consequential Parameter in Continuous Cellular Automata

## Introduction

Cellular automata (CA) dynamics with continuously-valued states and time steps can be generically written as[^note1]:

<img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/generic_cca.png">

In particular the equation above describes the Lenia framework for continuous CA, and as noted previously in that work [^Ch2019], the equation above has the same form as Euler's numerical method for solving differential equations, _i.e._ estimating CA dynamics if they are described by a differential equation <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/dat_over_dt.png" height=32>. CA updates under the Lenia framework are more particularly written as:

<img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/lenia.png">

Where <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/at_plus_dt.png" height=16> is the grid of cell states at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/t_plus_dt.png" height=16>, <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/growth_fn.png" height=16> is the growth function, <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/k_convolve_at.png" height=16> is the 2d spatial convolution of neighborhood kernel <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/k.png" height=16> with cell states <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/at.png" height=16> at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/t.png" height=16>, and <img src="https://raw.githubusercontent.com/riveSunder/yuca/step_size_pages/assets/equations/dt.png" height=16> is the step size. 

For numerical estimation of differential equations with [Euler's method](https://en.wikipedia.org/wiki/Euler_method), error depends on step size. That is to say that smaller step sizes lead to more accurate solutions. Those readers with familiarity in working with numerical methods for differential equations and/or physical simulations based on them will likely anticipate that a step size that is too large leads to unstable solutions and often, for simulations, catastrophic behavior. As an example, we can consider the following simulation of a robot drop in PyBullet [^pybullet] at varying step sizes. In the animation below, the bottom left corner (c) represents the default step size of 1/240 seconds, a reasonable trade-off between execution speed and simulation fidelity and stability. The top row a and b are based on the same initial conditions with step sizes 100 and 10 times smaller than c, respectively. The animation in d used a step size 100 times longer than the default step size in c. 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/pybullet_step_size.gif" width=50%><br>
<em>Modulating time_step parameter in PyBullet. <strong>a</strong>: dt = 1/24000 s, <strong>b</strong>: dt = 1/2400 s,<strong>c</strong>: dt = 1/240 s,<strong>d</strong>: dt = 1/2.4 s. 1/240 seconds is the default and recommended step size in PyBullet. </em>
</p>

Only the final condition with a step size of ~0.4166 seconds displays noticeably non-physical and unstable behavior: the two-wheeled robots plunge through the ground plane before rising slowly like so many spring flowers. Small step sizes behave qualitatively similar to the default (though specific outcomes differ). In continuous CA, similar differences in step size leads to compromised stability at too small a step size as well as too large. Specific patterns under otherwise identical CA rules occupy different ranges of stable step size. Patterns may also exhibit qualitatively different behaviors at different step sizes. Thus step size choice in continuous CA leads to more interesting differences in outcomes than we see in the PyBullet example and typically expect in numerical physics simulations in general. There are many other physics-based models that support self-organizing solitons, such as chemical reaction-diffusion models (_e.g._ [^Mu2014]), particle swarms (_e.g._ [^Sc2016]),  or other n-body problems, and these may also exhibit interesting behavioral diversity over a range of step sizes. 

## Pattern stability depends on step size

A minimal glider in the style of the 
Life glider [^rafler2012] and implemented in the _Scutium gravidus_ CA under the Lenia framework is only stable in a range of step sizes from about 0.25 to 0.97. A choice of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> outside this range results in a vanishing glider. 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/single_scutium.gif" width=50%><br>
</p>

It is worth noting that this glider in Lenia's _Scutium_ CA rules closely resembles the SmoothLife glider mentioned by Rafler [^rafler2012], where, because SmoothLife initially used a replacement instead of a residual update function, it was essentially simulated with a step size of 1.0. Modified to use the notation of Lenia, the discrete SmoothLife update can be written as: 

<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/smooth_life.png">

Where <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/s_fn.png" height=16> is the SmoothLife update function. While this discrete version of SmoothLife (in which the first glider was found) does not have a time step but the update function does depend on both the neighborhood states <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/k_convolve_at.png" height=16> _and_ the previous cell states <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/at.png" height=16>, a distinction discarded in Lenia. 

## Otherwise Identical CA May Support Different Patterns in Mutually Exclusive Step Size Ranges

A wide glider is typically stable for over 2000 steps at a <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/point_1.png">, but disappears at step sizes of 0.05 or below and exhibits unrestrained growth at a step size of 0.5.

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/superwide_scutium.gif" width=50%><br>
</p>

## Behavior of Individual Patterns May Vary Qualitatively at Different Step Size While Maintaining Self-Organization

A more striking consequence of step size is qualitatively different behavior at different step sizes. The following example is a "frog" pattern implemented in an extension of Lenia called Glaberish [^davis2022]. Earlier, our attention was drawn to the fact that in Lenia, only the results of a 2D convolution of the neighborhood kernel and the cell state grid is considered when computing the growth function <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/growth_fn.png">. Glaberish splits this growth function into _persistence_ and _genesis_ functions, each contributes to the overal change in cell state according to the current grid states.  

<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/glaberish.png">

Glaberish CA dynamics reinstate the dependence on cell state found in SmoothLife, Conway's Life [^gardner1970], and other Life-like CA, while maintaining the flexibility of the Lenia's growth function. The frog pattern can be found in a Glaberish CA with evolved persistence ang genesis parameters called s613 (see [^davis2022b] for details on how this CA was evolved). While the narrow and wide gliders in Lenia's _Scutium gravidus_ CA occupy particular (and mutually exclusive) ranges of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png">, the s613 frog pattern exhibits qualitatively different behavior across a range of step sizes from about 0.01 to about 0.13.  

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/supplemental_item_1_step_size_behavior.gif">
</p>


[^note1]: But note the different formulation for the original, discrete, SmoothLife, which had a discrete time-step (_i.e._ cell states were replaced at each time step): <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/smooth_life.png" height=18>
[^Ra2012]: Rafler, Stephan. “Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife.” arXiv: Cellular Automata and Lattice Gases (2011): [https://arxiv.org/abs/1111.1567](https://arxiv.org/abs/1111.1567)
[^Ch2019]: Chan, Bert Wang-Chak. “Lenia - Biology of Artificial Life.” Complex Syst. 28 (2019): [https://arxiv.org/abs/1812.05433](https://arxiv.org/abs/1812.05433)
[^Mu2014]: Munafo, Robert. “Stable localized moving patterns in the 2-D Gray-Scott model.” arXiv: Pattern Formation and Solitons (2014): [https://arxiv.org/abs/1501.01990](https://arxiv.org/abs/1501.01990)

[^Sc2016]: Schmickl, Thomas et al. “How a life-like system emerges from a simple particle motion law.” Scientific Reports 6 (2016): n. pag.
