
# Step Size is a Consequential Parameter in Continuous Cellular Automata

It should come as no surprise to anyone who has experience with a physics simulator that has a user-configurable time step that stability breaks down with a poor choice of this parameter. Typically, the breakdown occurs with a time step set too high, as systematic erros accumulate to push the system under simulation into unstable regimes. A larger time step offers faster simulation times, however, and so setting the parameter is a trade-off between simulation stability (and physical fidelity) against execution time and an efficient use of computational resources. Setting the step size too small usually only punishes the programmer with long run times, however, while stability and accuracy can be expected to at least meet the performance of a critically set step size parameter.

Continuous cellular automata (CA), particularly implemented in the Lenia framework, also have a configurable parameter <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> that determines how large an update to undertake at each update step. In fact, the formula for updating the CA grid takes the same form as Euler's method for numerical estimation of differential equations. 

<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/lenia.png"> 

Where <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/at_plus_dt.png">, the grid state at time step <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/t_plus_dt.png"> is determined by adding to the grid state at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/t.png"> the product of step size <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> and the results of a growth function <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/growth_fn.png"> applied to a 2D spatial convolution (<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/k_convolve_at.png">) of neighborhood kernel <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/k.png"> with the grid state at time <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/t.png">, <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/at.png">. As mentioned above, the Lenia update function resembles a very simple version (_i.e._ Euler's method) of a numerical method for solving a differential equation. More sophisticated differential equation solvers form the core of many physics simulators. Casual experimentation with one quickly yields the intuition that when one sets the step size too high, things fly apart (or otherwise act in a non-physical way). 

To set the stage for later observations of modulated step size in Lenia, let us consider the following simulation of a robot drop in PyBullet [^pybullet] at varying step sizes. In the animation below, the bottom left corner (c) represents the default step size of 1/240 seconds, a reasonable trade-off between execution speed and simulation fidelity and stability. The top row a and b are based on the same initial conditions with step sizes 100 and 10 times shorter the c, respectively. The animation in c used a step size 100 times longer than the default step size in c. 


<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/pybullet_step_size.gif">
<em>Modulating time_step parameter in PyBullet. <strong>a</strong>: dt = 1/24000 s, <strong>b</strong>: dt = 1/2400 s,<strong>c</strong>: dt = 1/240 s,<strong>d</strong>: dt = 1/2.4 s. 1/240 seconds is the default and recommended step size in PyBullet. </em>
</p>

Only the final condition with a step size of ~0.4166 seconds displays noticeably non-physical and unstable behavior, while a step size 10 to 100 times shorter than default behaves qualitatively similar to the default step size (though there are differences in specific outcomes). When similar swings in step size are applied to self-organizing patterns in Lenia and related continuous CA, we see that stability can be compromised at too small a step size as well as too large, and that specific patterns occupy different ranges of stable step size. Patterns may also exhibit qualitatively different behaviors at different step sizes. Thus step size choice in continuous CA leads to more interesting differences in outcomes than in the PyBullet example, though I would expect that with a more complex simulation in PyBullet (say 100s to 1000s of spheres with mutual interactions) there might be emergent phenomena that exhibit interesting differences in their activity as a consequence of step size choice. 

## Pattern Stability Depends on Step Size


A minimal glider in the style of the SmoothLife glider [^rafler2012] and implemented in the _Scutium gravidus_ CA under the Lenia framework is only stable in a range of step sizes from about 0.25 to 0.97. A choice of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> outside this range results in a vanishing glider. 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/single_scutium.gif">
</p>

It is worth noting that this glider in Lenia's _Scutium_ CA rules closely resembles the SmoothLife glider mentioned by Rafler [^rafler2012], where, because SmoothLife initially used a replacement instead of a residual update function, it was essentially simulated with a step size of 1.0. Modified to use the notation of Lenia, the discrete SmoothLife update can be written as: 

<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/smooth_life.png">

Where <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;s(\cdot)"> is the SmoothLife update function. While this discrete version of SmoothLife (in which the first glider was found) does not have a time step but the update function does depend on both the neighborhood states <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;K \ast A^t"> _and_ the previous cell states <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;A^t">, a distinction discarded in Lenia. 

## Otherwise Identical CA May Support Different Patterns in Mutually Exclusive Step Size Ranges

A wide glider is typically stable for over 2000 steps at a <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png"> of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/point_1.png">, but disappears at step sizes of 0.05 or below and exhibits unrestrained growth at a step size of 0.5.

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/superwide_scutium.gif">
</p>

## Behavior of Individual Patterns May Vary Qualitatively at Different Step Size While Maintaining Self-Organization

A more striking consequence of step size is qualitatively different behavior at different step sizes. The following example is a "frog" pattern implemented in an extension of Lenia called Glaberish [^davis2022]. Earlier, our attention was drawn to the fact that in Lenia, only the results of a 2D convolution of the neighborhood kernel and the cell state grid is considered when computing the growth function <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/growth_fn.png">. Glaberish splits this growth function into _persistence_ and _genesis_ functions, each contributes to the overal change in cell state according to the current grid states.  

<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/glaberish.png">

Glaberish CA dynamics reinstate the dependence on cell state found in SmoothLife, Conway's Life [^gardner1970], and other Life-like CA, while maintaining the flexibility of the Lenia's growth function. The frog pattern can be found in a Glaberish CA with evolved persistence ang genesis parameters called s613 (see [^davis2022b] for details on how this CA was evolved). While the narrow and wide gliders in Lenia's _Scutium gravidus_ CA occupy particular (and mutually exclusive) ranges of <img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/equations/dt.png">, the s613 frog pattern exhibits qualitatively different behavior across a range of step sizes from about 0.01 to about 0.13.  

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/consequential_step_size/supplemental_item_1_step_size_behavior.gif">
</p>


