
# Step Size is a Consequential Parameter in Continuous Cellular Automata

*Experiment with varying step size in an interactive* [notebook](https://github.com/riveSunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) *on*: [mybinder](https://mybinder.org/v2/gh/rivesunder/yuca/master?labpath=notebooks%2Fconsequential_step_size.ipynb) -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rivesunder/yuca/master?labpath=notebooks%2Fconsequential_step_size.ipynb) or in [colab](https://colab.research.google.com/github/rivesunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) -> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivesunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) 

## Introduction

Cellular automata (CA) dynamics with continuously-valued states and time steps can be generically written as[^note1]:

<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/generic_cca.png">

In particular the equation above describes the Lenia framework for continuous CA, and as noted previously in that work [^Ch2019], the equation above has the same form as Euler's numerical method for solving differential equations, _i.e._ estimating CA dynamics if they are described by a differential equation <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/dat_over_dt.png" height=32>. CA updates under the Lenia framework are more particularly written as:

<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/lenia.png">

Where <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/at_plus_dt.png" height=16> is the grid of cell states at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/t_plus_dt.png" height=16>, <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/growth_fn.png" height=16> is the growth function, <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/k_convolve_at.png" height=16> is the 2d spatial convolution of neighborhood kernel <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/k.png" height=16> with cell states <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/at.png" height=16> at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/t.png" height=16>, and <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/dt.png" height=16> is the step size. 

For numerical estimation of differential equations with [Euler's method](https://en.wikipedia.org/wiki/Euler_method), error depends on step size. That is to say that smaller step sizes lead to more accurate solutions. Those readers with familiarity in working with numerical methods for differential equations and/or physical simulations based on them will likely anticipate that a step size that is too large leads to unstable solutions and often, for simulations, catastrophic behavior. As an example, we can consider the following simulation of a robot drop in PyBullet [^pybullet] at varying step sizes. In the animation below, the bottom left corner (c) represents the default step size of 1/240 seconds, a reasonable trade-off between execution speed and simulation fidelity and stability. The top row a and b are based on the same initial conditions with step sizes 100 and 10 times smaller than c, respectively. The animation in d used a step size 100 times longer than the default step size in c. 

{:style="text-align:center;"}
![Image](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/consequential_step_size/pybullet_step_size.gif)

{:refdef: style="text-align: center;"}
**Modulating time_step parameter in PyBullet. <strong>a</strong>: dt = 1/24000 s, <strong>b</strong>: dt = 1/2400 s,<strong>c</strong>: dt = 1/240 s,<strong>d</strong>: dt = 1/2.4 s. 1/240 seconds is the default and recommended step size in PyBullet.**
{: refdef}

Only the final condition with a step size of ~0.4166 seconds displays noticeably non-physical and unstable behavior: the two-wheeled robots plunge through the ground plane before rising slowly like so many spring flowers. Small step sizes behave qualitatively similar to the default (though specific outcomes differ). In continuous CA, similar differences in step size leads to compromised stability at too small a step size as well as too large. Specific patterns under otherwise identical CA rules occupy different ranges of stable step size. Patterns may also exhibit qualitatively different behaviors at different step sizes. Thus step size choice in continuous CA leads to more interesting differences in outcomes than we see in the PyBullet example and typically expect in numerical physics simulations in general. There are many other physics-based models that support self-organizing solitons, such as chemical reaction-diffusion models (_e.g._ [^Mu2014]), particle swarms (_e.g._ [^Sc2016]),  or other n-body problems, and these may also exhibit interesting behavioral diversity over a range of step sizes. 

## Pattern stability depends on step size

A minimal glider in the style of the 
Life glider [^Ra2012] and implemented in the _Scutium gravidus_ CA under the Lenia framework is only stable in a range of step sizes from about 0.25 to 0.97. A choice of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> outside this range results in a vanishing glider. 

{:style="text-align:center;"}
![Image](https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/single_scutium.gif)

{:refdef: style="text-align: center;"}
**A minimal glider in Lenia's _Scutium gravidus_ rule set [^Ch2019], similar to the SmoothLife glider [^Ra2012], is unstable at step sizes below about 0.25 and above about 0.97.**
{: refdef}

A wide glider is typically stable for over 2000 steps at a <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png" height=16> of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/point_1.png" height=16>, but disappears at step sizes of 0.05 or below and is also unstable at a step size of 0.5 or above, usually exhibiting unconstrained growth at large step sizes.

{:style="text-align:center;"}
![Image](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/consequential_step_size/superwide_scutium.gif)

{:refdef: style="text-align: center;"}
**A wide glider in _Scutium gravidus_. Unlike the narrow glider, this glider is pseudo-stable at a moderate step size of 0.1 and unstable for large and small step sizes above and below about 0.5 and 0.05, respectively.**
{: refdef}

## Behavior of individual patterns can vary qualitatively at different step sizes

A more striking consequence of step size is qualitatively different behavior at different step sizes. The following example is a "frog" pattern implemented in an extension of Lenia called Glaberish [^Da2022]. In Lenia, the growth function <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/growth_fn.png" height=16>  depends only on the results of a 2D convolution of the neighborhood kernel and the cell state grid <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/k_convolve_at.png" height=16>. Glaberish splits this growth function into _persistence_ (<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/persistence_fn.png" height=16>) and _genesis_ (<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/genesis_fn.png" height=16>) functions, each contributes to the overal change in cell state according to the current grid states.  

<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/glaberish.png">

Glaberish CA dynamics reinstate the dependence on cell state found in SmoothLife, Conway's Life [^Ga1970], and other Life-like CA, while maintaining the flexibility of Lenia's growth function. The following frog pattern can be found in a Glaberish CA with evolved persistence ang genesis parameters called s613 (see [^Da2022b] for details on how this CA was evolved). While the narrow and wide gliders in Lenia's _Scutium gravidus_ CA occupy particular ranges of <img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/dt.png" height=16>, the s613 frog pattern exhibits qualitatively different behavior across a range of step sizes from about 0.01 to about 0.13.  

{:style="text-align:center;"}
![Image](https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/supplemental_item_1_step_size_behavior.gif)

{:refdef: style="text-align: center;"}
**For the frog pattern in Glaberish CA s613, varying step size leads to qualitatively different behaviors.**
{: refdef}

## Conclusions

This work demonstrates the consequences of varying step size in continuous CA. Patterns simulated in Lenia's _Scutium gravidus_ CA are unstable at either too large or too small step size, and different patterns occupy different step size ranges in otherwise identical CA rules. In the Glaberish CA s613, the frog pattern exhibits qualititatively different behavior at step sizes from 0.01 to 0.15, ranging from corkscrewing, meandering, hopping, surging, and finally bursting and disappearing. 

The results we have observed for these patterns contrasts sharply with previous remarks concerning the similarity of continuous CA to Euler's method for solving ODEs with regard to step size [^Ch2019]. Observations of the mobile _Orbium_ pattern in Lenia were consistent with the premise that decreasing step size asymptotically approaches an ideal simulation of the _Orbium_ pattern [^Ch2019], but for gliders in _Scutium gravidus_ and s613 we have shown that the relationship between CA dynamics and step size is not that simple in general. This work demonstrates that for several patterns a lower step size does not entail a more accurate simulation, but different behavior or potential patterns entirely. Given the evidence presented in this work, it follows that step size should be given due consideration when searching for bioreminiscent patterns [^Ch2019] [^Ch2020], and for optimization and learning with CA, for example in training patterns to have the agency to negotiate obstacles [^Ha2022], or for training neural CA for a variety of tasks such as growing patterns [^Mo2020], classifying pixels [^Ra2020], learning to generate textures [^Ni2021], and control [^Va2021].

*Experiment with varying step size in an interactive* [notebook](https://github.com/riveSunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) *on*: [mybinder](https://mybinder.org/v2/gh/rivesunder/yuca/master?labpath=notebooks%2Fconsequential_step_size.ipynb) -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rivesunder/yuca/master?labpath=notebooks%2Fconsequential_step_size.ipynb) or in [colab](https://colab.research.google.com/github/rivesunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) -> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivesunder/yuca/blob/master/notebooks/consequential_step_size.ipynb) 

**This post has not been peer-reviewed itself, but provides supporting information for the following short article accepted to the 2022 Conference on Artificial Life:**

* Davis, Q, Tyrell and Bongard, Josh. "Step Size is a Consequential Parameter in Continuous Cellular Automata". Accepted to The 2022 Conference on Artificial Life. (2022).

## References and Footnotes

[^note1]: **Note:** The different formulation for the original, discrete, SmoothLife, which had a discrete time-step (_i.e._ cell states were replaced at each time step): <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/smooth_life.png" height=18>


[^Ch2019]: Chan, Bert Wang-Chak. "Lenia - Biology of Artificial Life." Complex Syst. 28 (2019): [https://arxiv.org/abs/1812.05433](https://arxiv.org/abs/1812.05433).

[^Ch2020]: Chan, Bert Wang-Chak. "Lenia and Expanded Universe." ALIFE 2020: The 2020 Conference on Artificial Life. MIT Press, (2020). [https://arxiv.org/abs/2005.03742](https://arxiv.org/abs/2005.03742)

[^Da2022]: Davis, Q, Tyrell and Bongard, Josh. "Glaberish: Generalizing the Continuously-Valued Lenia Framework to Arbitrary Life-Like Cellular Automata". Accepted to The 2022 Conference on Artificial Life. (2022).

[^Da2022b]: Davis, Q, Tyrell and Bongard, Josh. "Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis". Accepted to Proceedings of the Genetic and Evolutionary Computation Conference Companion (2022): [https://arxiv.org/abs/2204.07541](https://arxiv.org/abs/2204.07541)

[^Ga1970]: Gardener, Master. "Mathematical games: the fantastic combinations of john conway's new solitaire game 'life'." (1970).

[^Ha2022]: Hamon, G. and Etcheverry, M. and Chan, B. W.-C. and Moulin-Frier, C. and Oudeyer, P.-Y. Learning sensorimotor agency in cellular automata. Blog post. (2022). [https://developmentalsystems.org/sensorimotor-lenia/](https://developmentalsystems.org/sensorimotor-lenia/).

[^Mo2020]: Mordvintsev, A. and Randazzo, E. and Niklasson, E. and Levin, M. Growing neural cellular automata. Distill. (2020) [https://distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca).

[^Mu2014]: Munafo, Robert. "Stable localized moving patterns in the 2-D Gray-Scott model." arXiv: Pattern Formation and Solitons (2014): [https://arxiv.org/abs/1501.01990](https://arxiv.org/abs/1501.01990).

[^Ni2021]: Niklasson, E. and Mordvintsev, A. and Randazzo, E. and Levin, M. "Self-Organising Textures", Distill, (2021). [https://distill.pub/selforg/2021/textures/](https://distill.pub/selforg/2021/textures/)

[^Ra2012]: Rafler, Stephan. “Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife.” arXiv: Cellular Automata and Lattice Gases (2011): [https://arxiv.org/abs/1111.1567](https://arxiv.org/abs/1111.1567)

[^Ra2020]: Randazzo, E. and Mordvintsev, A. and Niklasson, E. and Levin, M. and Greydanus, S. Self-classifying MNIST digits. Distill. (2020) [https://distill.pub/2020/selforg/mnist](https://distill.pub/2020/selforg/mnist).

[^Sc2016]: Schmickl, Thomas et al. "How a life-like system emerges from a simple particle motion law." Scientific Reports 6 (2016): [https://www.nature.com/articles/srep37969](https://www.nature.com/articles/srep37969)

[^Va2021]: Variengien, A. and Pontes-Filho and Sidny abd Glover, T. and Nichele, S. "Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent." Innovations in Machine Intelligence, 1:1–14. (2021). [https://arxiv.org/abs/2106.15240v1](https://arxiv.org/abs/2106.15240v1)
