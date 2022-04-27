# Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/s3_waiting_sedentary_pattern.gif">
</p>

# Introduction

<blockquote>
If you wish to make an apple pie from scratch, you must first invent the universe. ---Carl Sagan _Cosmos_ 1980 
</blockquote> 

[^Sa1980]

Lucky for pie-lovers everywhere, we already have a universe with all the physical laws and pre-requisites for the existence of a wide range of baked goods and bakers to make them. That's convenient, because universe creation is not in our experimental repertoire, at least for the type of universe we reside in. We can, however, reason about and experiment with general principles for the emergence of life-like or bioreminiscent characteristics in simplified, artificial worlds, and in this work we focus on continuously-valued cellular automata (CA) as our substrate. 

Recent work in continuous CA, in particular in the Lenia framework [^Ch2018a], continues to generate a vast taxonomy of bioreminiscent patterns. To date this has mostly been the product of manual exploration and human-directed interactive evolution [^Ch2018][^Ch2020], resulting in 100s of bioreminiscent patterns (I'll occasionally refer to these as pseudorganisms here) that move, interact, and maintain self-integrity to some degree. Limiting exploration of continuous CA to manual or semi-automated methods may limit the diversity of results [^note1], which may in turn limit the ideas considered as falsifiable hypotheses as the study of ALife in continuous CA matures. 

I applied the evolutionary methods described here not as a replacement for, but a complement to, manual/semi-automated approaches. Not intending to replace manual and semi-automated approaches to exploration. Given the myriad constructions and discoveries in discrete CA, largely the product of human tinkering, it would be foolish to discount human-in-the-loop methods [^note2]. This work was also motivated by a desire to submit to an evolution-themed conference.

# Background

When Carl Sagan spoke of inventing a universe to make an apple pie he was principally talking about the construction of heavier elements from hydrogen, a process that takes place in the fusion cores of stars. Without a universe that supports stellar lives like our own, their would be no stars, and without stars we shouldn't expect the ingredients of a typical pie recipe to exist. But another requisite for making apple pies is a universe that can support the bakers.

In the great demotions (another Sagan-ism) kicked off with the Copernican revolution, humanity collectively came to realize that we are not, in fact, the center of the universe. (_*Examples*_). Coupled with the discovery of evolution, one may be tempted to conclude that the human perspective is not only not special, but absolutely typical of the way of things in the Cosmos. As a counter to this conclusion, which would lead us to expect a universe teeming with unmistakable civilizations just like our own, Brandon Carter introduced the anthropic principle [^Ca1974][^Ca1984]. There are two types of anthropic principle (and many variations thereof), the weak anthropic principle asserts that while sapient life like humans cannot expect to be central, it must recognize that its own perspective is necessarily privileged by the conditions required for its existence. A strong anthropic principle, on the other hand, often invokes a phrase in the style of Descartes: "cogito ergo mundus talis est," or "I think, therefore the world is such as it is [^Ca1974]. <!-- More extreme strong anthropic principles (of the type espoused by Barrow and Tipler) can lead to assuming a universe that _must_ produce intelligent life, that is brought into existence only by perception by intelligent observers contained therein, the necessity of a multiverse, and so on. --> 

Strong anthropic principles typically suggest the universe is fine-tuned for the existence of intelligent life, a premise for which Carter invokes the concept of world-ensembles: a set of universes occupying all possibe initial conditions and global laws. From these, it may be posited, only those that can support organisms that can act as observers, such as ours, will contain observers (anthropic principles are tautological) and perhaps with some philosophical flexibility one can guess at the prevalence of universes with necessary conditions for life. From an evolutionary perspective, which we adopt for this work, we can stretch the idea of anthropic principles to say that weak principles are characterized by life selected for by the constraints of their universe of residence, and strong principles invoke that universes are selected by their ability to produce life that may act as observers. These ideas are not particularly amenable to direct experimentation, but we can apply them as mental tools in considering the occurence of artificial life in physically consistent simulated universes, such as cellular automata (CA). We'll continue to use the abbreviation CA for both singular (cellular automaton) and plural instances.  

Von Neumann's universal construction CA is the archetypal example of a complex system that is carefully engineered for a specific purpose [^Vo1996]. The Von Neumann CA has 29 states, including a ground state, 8 transition states, 4 confluent states, 4 ordinary transmission states, and 4 special transmission states. The various states, which indicate direction as well as type, follow a complicated list of update rules. John H. Conway, on the other hand, supposed that detailed engineering was not strictly necessary to build a CA capable of interesting complexity. 


# Evolved Lenia Zoo

In general, the rule sets described in the Lenia framework are named for the patterns they support (see [^Ch2018a] and especially the interactive demo [^Ch2018b]] for examples), and these were developed in tandem via manual manipulation and interactive evolution. An automated evolutionary approach should at least be able to select for gliders in the Lenia CA that are already known to support them, even where the specific patterns may be different than the Lenia originals. 

## Re-discoveries

While some of the glider patterns evolved in Lenia CA were apparently not previously documented, quite a few are the same as the patterns described in [^Ch2018a][^Ch2018b]. We can view these re-discoveries below. 

### _Orbium bicaudatus_

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/zoo/lenia_zoo/discutium_valvatus_glider_00.gif">
</p>

_Orbium bicaudatus_ is a Lenia rule set with <img src="https://render.githubusercontent.com/render/math?math=\mu = 0.15"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma = 0.014">. It uses the _Orbium_ neighborhood kernel with <img src="https://render.githubusercontent.com/render/math?math=\mu_k = 0.5"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_k = 0.15">. 
 

### _Hydrogemium natans_ 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/zoo/lenia_zoo/hydrogeminium_natans_glider_01.gif">
</p>

_Hydrogeminium natans_ is a Lenia rule set with an update rule defined by $$\mu = 0.26$$ and $$\sigma = 0.036$$ and a neighborhood with three rings centered at $$\mu_{k_n} = (0.0938, 0.2814, 0.4690)$$, with $$\sigma_{k_n} = (0.033, 0.033, 0.033)$$ and weighted by $$(0.5, 1.0, 0.667)$$. 

## Putatively new gliders in Lenia CA

### _Discutium solidus_ 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/zoo/lenia_zoo/discutium_solidus_glider_00.gif">
</p>

_Discutium solidus_ is a Lenia rule set with <img src="https://render.githubusercontent.com/render/math?math=\mu = 0.356"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma = 0.063">. 

### _Discutium valvatus_ 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/zoo/lenia_zoo/discutium_valvatus_glider_00.gif">
</p>

_Discutium valvatus_ is a Lenia rule set with <img src="https://render.githubusercontent.com/render/math?math=\mu = 0.337"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma = 0.0595">. 

### _Hydrogemium natans 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/gecco_pages/assets/zoo/lenia_zoo/hydrogeminium_natans_glider_00.gif">
</p>

_Hydrogeminium natans_ is a Lenia rule set with an update rule defined by <img src="https://render.githubusercontent.com/render/math?math=\mu = 0.26"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma = 0.036"> and a neighborhood with three rings centered at <img src="https://render.githubusercontent.com/render/math?math=\mu_{k_n} = (0.0938, 0.2814, 0.4690)">, with <img src="https://render.githubusercontent.com/render/math?math=\sigma_{k_n} = (0.033, 0.033, 0.033)"> and weighted by <img src="https://render.githubusercontent.com/render/math?math=(0.5, 1.0, 0.667)">. The glider shown here has the same behavior (but different cell values) than fast/wobbly gliders found in s613 and s643 evolved CA, which share the _Hydrogeminium_ neighborhood kernel. 


[^note1]: Consider the gap in diversity of domestic versus wild life. 
[^note2]: See for example [conwaylife.com forums](https://conwaylife.com/forums/viewtopic.php?f=11&t=2597) or the built-in demos in CA simulation software [Golly](https://conwaylife.com/wiki/Golly) [^Golly2016].
[^Vo1966]: Neumann, John von and Arthur W. Burks. "Theory Of Self Reproducing Automata." University of Illinois Press, Urbana and London. (1966).
[^Ca1974]: Carter, Brandon. "Large number coincidences and the anthropic principle in cosmology." Confrontation of cosmological theories with observational data. Springer, Dordrecht, 1974. 291-298.
[^Ca1984]: Carter, Brandon. "The anthropic principle and its implications for biological evolution." Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 310.1512 (1983): 347-363.
[^Sa1980]: Carl Sagan. Cosmos. Random House, New York. 1980. ISBN: 0-394-50294-9 p. 218
[^Ch2018a]:B. W.-C. Chan, "Lenia: Biology of Artificial Life," Complex Systems, 28(3), 2019 pp. 251–286. https://doi.org/10.25088/ComplexSystems.28.3.251
[^Ch2018b]: [https://chakazul.github.io/Lenia/JavaScript/Lenia.html](https://chakazul.github.io/Lenia/JavaScript/Lenia.html)
[^Golly2016]: Trevorrow, A., Rokicki, T., Hutton, T., Greene, D., Summers, J., Verver, M., Munafo, R., and Rowett, C. Golly version 2.8. (2016).
