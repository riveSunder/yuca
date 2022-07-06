# Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis
<p align="center">
Q. Tyrell Davis and Josh Bongard 

ArXiv -> [https://arxiv.org/abs/2204.07541](https://arxiv.org/abs/2204.07541)
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/frog_race.gif" width=50%>
</p>

## Summary

<blockquote>
If you wish to make an apple pie from scratch, you must first invent the universe. ---Carl Sagan _Cosmos_ 1980 
</blockquote> 

[^Sa1980]

Lucky for pie-lovers everywhere, we already have a universe with all the physical laws and pre-requisites for the existence of a wide range of baked goods and bakers to make them. That's convenient, because universe creation is not in our experimental repertoire, at least for the type of universe we reside in. We can, however, reason about and experiment with general principles for the emergence of life-like or bioreminiscent characteristics in simplified, artificial worlds, and in this work we focus on continuously-valued cellular automata (CA) as our substrate. 

Recent work in continuous CA, in particular in the Lenia framework [^Ch2018a], has generated a vast taxonomy of bioreminiscent patterns. To date this has mostly been the product of manual exploration and human-directed interactive evolution [^Ch2018][^Ch2020], resulting in 100s of life-like patterns (or "pseudorganisms") that move, interact, and maintain self-integrity to some degree. Limiting exploration of continuous CA to manual or semi-automated methods may limit the diversity of results [^note1], which may in turn limit the ideas considered as falsifiable hypotheses as the study of ALife in continuous CA matures. 

Complicated and carefully engineered CA systems in the tradition of John Von Neumann's universal constructor CA [^Vo1966] can display life-like characteristics such as self-replication, movement, and growth. On the other hand Conway's Game of Life [^Be2004] and successors showed that very simple CA rules can give rise to complex systems with similar capabilities, and it seems that only minimal criteria need to be met in a simple complex system to achieve life-like traits and computational capability. 

Unlike Von Neumann's 29-state CA, we can describe the development of Conway's Life as evolution via selection for human preferences. One of the selection criteria that emerged under and encompassing search for something interesting was the simultaneous support for opposing capabilities to grow without bound or to vanish completely. In fact the inability to predict whether a given pattern under a given set of CA ruleswill persist or vanish is in fact a version of the halting _Entscheidungensproblem_ (decision problem).

[![John Horton Conway](https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/john_h_conway_2005_quote)](https://www.youtube.com/watch?v=R9Plq-D1gEk&t=490s) 

_In the article introducing the public to Conway's Game of Life (In Martin Gardner's 'Mathematical Games' column of Scientific American [^Ga1970]), a prize was offered for the first proof of a pattern in Life that exhibits indefinite growth. Quote is from a 2011 interview with John Conway by Dierk Schleicher [^Sc2011]. Image is adapted from [photograph CC BY Thane Plambeck](https://www.flickr.com/photos/thane/20366806)_

The casual heuristic of persistent and vanishing patterns that Conway and colleagues employed become the basis for Eppstein's _fertility_ and _mortality_ metrics. Eppstein's treatment was even more lenient in that it suggests that any Life-like CA that has one or more patterns that grow outside initial bound (feritlity) and one or more patterns that disappear (mortality) is likely complex enough to support universal computation [^Ep2010][^note3].

Working in the substrate of continuous CA (Lenia and a variant called Glaberish), in this work we automatically evolved CA rules, followed by a second stage of evolution to evolve patterns within the new rule sets. We applied selection via halting unpredictability (using a new ensemble of 3 conv-nets to try to learn halting prediction for each CA rule candidate. We also used a simpler selection criteria based on an even proportion of persistent and quiescent grids after a specified number of update steps, starting from a grid of random uniform cell states. Compared to random samples from the starting distributions of these CA rule sets (starting with values near those of the _Hydrogeminium natans_ and _Orbium_ Lenia rules), both halting unpredictability and halting proportion evolution yields more rule sets that support persistent gliders in a subsequent pattern evolution step. Given the additional computational resources of training an ensemble of neural networks for halting prediction vs simply counting the number of grids with and without live cells at the end, it seems that selecting for the simple existence of halting and growth patterns under typical circumstances is preferable to evolving halting unpredictability.

This blog post is a description of work presented as a [poster](https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/pos237s1.pdf) and more thoroughly as an accompanying [short paper](https://arxiv.org/abs/2204.07541) at [GECCO 2022](https://gecco-2022.sigevo.org/). A library called Your Universal Cellular Automata (yuca) was developed for and used to run experiments, and you can evolve your own CA rules and patterns using the open sourced MIT-licensed [code](https://github.com/rivesunder/yuca). 

The work received funding from the National Science Foundation under the Emerging Frontiers in Research and Innovation (EFRI) program (EFMA-1830870)

## Visual methods summary

### Phase one: evolving CA rules

In this first phase, CA rules were evolved according to selection for halting unpredictability or roughly equal halting/persistence proportions in a batch of grids initialized from a random uniform distribution. We'll refer to the first case as halting unpredictability evolution and the second a simple halting evolution. 

'Halting unpredictability' was based on the (negative) average accuracy of an ensemble of convolutional neural networks trained to predict whether a CA pattern would persist or vanish (all cells go to zero) after a given number of CA steps. Simple halting was instead based on the total proportion of persistent versus vanished grids after a given number of time steps, specifically the difference between the proportion of persistent/vanished grids and an even 0.5/0.5 split. 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/halting_prediction_summary.png" width=50%>
</p>

An example command to run evolution with fitness based on halting unpredictability is

```
python -m yuca.evolve -b 64 -c 512 -d cuda -f HaltingWrapper -g 32 -p 64 -k 31 -l 3 -m 128 -s 42
```

The command above specifies (in order of flags) a batch size of 64, 512 CA steps, to run on GPU, halting unpredictability-based fitness, 32 generations, a population size of 64, 3 replicates (_i.e._ train 3 separate conv-nets per each episode), a grid dimension of 128 by 128, and a random seed of 42. 

Using simple halting fitness is more economical in terms of computational resources, and seems to generate just as many glider-supporting CA as the more convoluted halting unpredictability fitness. The command is nearly the same: 

```
python -m yuca.evolve -b 64 -c 512 -d cuda -f SimpleHaltingWrapper -g 32 -p 64 -k 31 -l 3 -m 128 -s 42
```

### Phase two: evolving gliders

Gliders (here referring to mobile CA patterns in general)  act as information carriers and can carry out computations in their collisions, and can be quite charismatic and/or aesthetically pleasing in their activity. A [recent project](https://developmentalsystems.org/sensorimotor-lenia/) described the nascent training of agency in Lenia-based gliders, which may provide the substrate for studying agency and learning under consistent physics (_i.e._ a glider operates under the same rules as its environment.  

The glider evolution step in this project uses the same covariance matrix adaptation evolution strategy [^Ha2012] as for evolving CAs, coupled with a fitness metric comprised of motility and homeostasis components and compositional pattern-producing networks for encoding starting synthesis patterns [^St2007]. The motility component is calculated by finding the "center-of-mass" of active cells in the pattern, producing a positive reward when its position changes. A homeostasis component is a negative reward based on changes in the average cell value of all cells in the grid. Combined these metrics provide increased reward for patterns that move across the grid without growing too much or disappearing. There is also a large negative penalty for patterns that disappear before the simulation completes its run. 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/cppn_flow.png" width=50%>
</p>

Glider evolution has the same entry point but uses a different reward wrapper. There's no point 
in setting the replicates flag `-l` to a value other than 1, because each CPPN individual produces a static starting synthesis pattern. 

```
python -m yuca.evolve -b 64 -c 512 -d cuda -f GliderWrapper -g 32 -p 64 -k 31 -l 1 -m 128 -s 42
```

## Appendix 1: Re-discovered pattern zoo (Lenia patterns)

Most of the glider patterns evolved in Lenia CA were previously documented in [^Ch2018a], [^Ch2020], and in an online interactive demo: [chakazul.github.io/Lenia/JavaScript/Lenia.html](https://chakazul.github.io/Lenia/JavaScript/Lenia.html).


### _Hydrogeminium natans_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_gemini_pattern_101_103_107_109_1643777919_end_101_elite0_0278.gif" width=50%>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/hydrogeminum_natans_cucumberiform.gif" width=50%>
</p>


### _Discutium solidus_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_discutium_solidus_pattern_1645113835_end_107_elite0_0777.gif" width=50%>
</p>


### _Discutium valvatus_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_discutium_valvatus_pattern_1645176529_end_107_elite3_0636.gif" width=50%>
</p>

### _Scutium gravidus_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_scutium_gravidus_pattern_1645188721_end_107_elite4_0271.gif" width=50%>
</p>

### _Scutium solidus_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_scutium_solidus_pattern_1645085384_end_101_elite2_0483.gif" width=50%>
</p>

### _Scutium valvatus_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_scutium_valvatus_pattern_1645188440_end_101_elite5_0525.gif" width=50%>
</p>

### _Paraptera sinus labens_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_p_sinus_labens_pattern_1645194743_end_107_elite0_0288.gif" width=50%>
</p>

### _Paraptera arcus labens_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_p_arcus_labens_pattern_1649914682_end_11_elite0_0518.gif" width=50%>
</p>

### _Orbium_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_orbium_pattern_1645177361_end_107_elite1_0657.gif" width=50%>
</p>

### _Orbium unicaudatus ignis_ 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_o_bicaudatus_ignis_pattern_1645072355_end_103_elite3_0472.gif" width=50%>
</p>

### _Synorbium_ 


<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/lenia/gif_exp_synorbium_pattern_1645177716_end_107_elite0_0592.gif" width=50%>
</p>

## Appendix 2: New patterns evolved in evolved CA

This section includes a selection of glider patterns that don't resemble Lenia patterns, evolved under CA rules that were in turn evolved for halting/persistence or halting unpredictability. Other evolved rules yielded gliders as well, but with patterns that generally resembled previously described Lenia patterns (espeically _Orbia_) 


### Evolved CA s613 (halting unpredictability selection) 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/evolved/gif_exp_s613_pattern_1645038142_end_107_elite0_0991.gif" width=50%>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/evolved/s613_fast_glider.gif" width=50%>
</p>

### Evolved CA s643 (Simple halting/persistence selection) 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/evolved/gif_exp_simevgeminium_643_pattern_101_103_1643874810_end_101_elite0_0902.gif" width=50%>
</p>


### Evolved CA s11 (Simple halting/persistence selection) 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/evolved/s11_slow.gif" width=50%>
</p>


### Unevolved CA (random selection) 

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_2022_pages/assets/halting_evo/evolved/gif_rando_morpho_0955.gif" width=50%>
<br>
This CA rule set was "unevolved" <em>i.e.</em> instead of selection for halting/persistence or halting unpredictability, fitness was assigned at random. Nonetheless the rule set was able to support the pseudo-glider pattern shown above, evolved with the same center-of-mass and homeostasis selection mechanisms as the gliders found in evolved CA rule sets. The pattern is only pseudo-stable, however, and undergoes several shape changes before breaking down. 
</p>


[^note1]: Consider the gap in diversity of domestic versus wild life. 

<!-- [^note2]: See for example [conwaylife.com forums](https://conwaylife.com/forums/viewtopic.php?f=11&t=2597) or the built-in demos in CA simulation software [Golly](https://conwaylife.com/wiki/Golly) [^Golly2016].-->

[^note3]: There are exceptions that do not meet both criteria and yet are still capable of universal computation and interesting activity, many of which are mentioned in Eppstein's paper. One example, Life without Death (B3/S012345678), is the antithesis of a mortal CA but does support computationally complete structures uses a kind of rod logic (_e.g._ [https://conwaylife.com/forums/viewtopic.php?t=&p=106546#p106546](https://conwaylife.com/forums/viewtopic.php?t=&p=106546#p106546) )


[^Vo1966]: Neumann, John von and Arthur W. Burks. "Theory Of Self Reproducing Automata." University of Illinois Press, Urbana and London. (1966).
[^Ca1974]: Carter, Brandon. "Large number coincidences and the anthropic principle in cosmology." Confrontation of cosmological theories with observational data. Springer, Dordrecht, 1974. 291-298.
[^Ca1984]: Carter, Brandon. "The anthropic principle and its implications for biological evolution." Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 310.1512 (1983): 347-363.
[^Sa1980]: Carl Sagan. Cosmos. Random House, New York. 1980. ISBN: 0-394-50294-9 p. 218
[^Ch2018a]:B. W.-C. Chan, "Lenia: Biology of Artificial Life," Complex Systems, 28(3), 2019 pp. 251–286. https://doi.org/10.25088/ComplexSystems.28.3.251
[^Ch2020]: Chan, Bert Wang-Chak. "Lenia and Expanded Universe." ALIFE 2020: The 2020 Conference on Artificial Life. MIT Press, 2020. [https://arxiv.org/abs/2005.03742](https://arxiv.org/abs/2005.03742)
[^Golly2016]: Trevorrow, A., Rokicki, T., Hutton, T., Greene, D., Summers, J., Verver, M., Munafo, R., and Rowett, C. Golly version 2.8. (2016).
[^Ep2010]: Eppstein, David. "Growth and decay in life-like cellular automata." Game of Life cellular automata. Springer, London, 2010. 71-97. [https://arxiv.org/abs/0911.2890](https://arxiv.org/abs/0911.2890)
[^Ha2012]: Auger, Anne, and Nikolaus Hansen. "Tutorial CMA-ES: evolution strategies and covariance matrix adaptation." Proceedings of the 14th annual conference companion on Genetic and evolutionary computation. 2012.
[^St2007]: Stanley, Kenneth O. "Compositional pattern producing networks: A novel abstraction of development." Genetic programming and evolvable machines 8.2 (2007): 131-162.
[^Ga1970]: Gardner, Martin. "The Fantastic Combinations of Jhon Conway's New Solitaire Game'Life." Sc. Am. 223 (1970): 20-123.
[^Sc2011]: Dierk Schleicher. Interview with John Horton Conway. 2011 [https://www.ams.org/notices/201305/rnoti-p567.pdf](https://www.ams.org/notices/201305/rnoti-p567.pdf)
