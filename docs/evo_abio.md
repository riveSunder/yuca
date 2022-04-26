# Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/s3_waiting_sedentary_pattern.gif">
</p>

# Exploring our place in our universe by making new ones 

<blockquote>
If you wish to make an applie pie from scratch, you must first invent the universe. ---Carl Sagan _Cosmos_ 1980[^Sa1980] 
</blockquote>

In one of many famous quotes from Carl Sagan, when he spoke of inventing a universe in order to make an apple pie he was principally talking about the construction of heavier elements from hydrogen, a process that takes place in the fusion cores of stars. Without a universe that supports stellar lives like our own, their would be no stars, and without stars we shouldn't expect the ingredients of a typical pie recipe to exist. But another requisite for making apple pies is a universe that can support the bakers.

In the 'great demotions' (another Sagan-ism) kicked off with the Copernican revolution, humanity collectively came to realize that we are not, in fact, the center of the unvierse. (_*Examples*_). Coupled with the discovery of evolution, one may be tempted to conclude that the human perspective is not only not special, but absolutely typical of the way of things in the Cosmos. As a counter to this conclusion, which would lead us to expect a universe teeming with unmistakable civilizations just like our own, Brandon Carter introduced the anthropic principle [^Ca1974][^Ca1984].



# Evolved Lenia Zoo

In general, the rule sets described in the Lenia framework are named for the patterns they support (see [^Ch2019a] and especially the interactive demo [^Ch2019b]] for examples), and these were developed in tandem via manual manipulation and interactive evolution. An automated evolutionary approach should at least be able to select for gliders in the Lenia CA that are already known to support them, even where the specific patterns may be different than the Lenia originals. 

## Re-discoveries

While some of the glider patterns evolved in Lenia CA were apparently not previously documented, quite a few are the same as the patterns described in [^Ch2018a][^Ch2018b]. We can view these re-discoveries below. 

### _Orbium bicaudatus_

<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/zoo/lenia_zoo/discutium_valvatus_glider_00.gif">
</p>

_Orbium bicaudatus_ is a Lenia rule set with $\mu = 0.15$ and $\sigma = 0.014$. It uses the _Orbium_ neighborhood kernel with $\mu_k = 0.5$ and $\sigma_k = 0.15$. 
 
## Putatively new gliders in Lenia CA

### _Discutium solidus_ 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/zoo/lenia_zoo/discutium_solidus_glider_00.gif">
</p>

_Discutium solidus_ is a Lenia rule set with $\mu = 0.356$ and $\sigma = 0.063$. 

### _Discutium valvatus_ 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/zoo/lenia_zoo/discutium_valvatus_glider_00.gif">
</p>

_Discutium valvatus_ is a Lenia rule set with $\mu = 0.337$ and $\sigma = 0.0595$. 

### _Hydrogemium natans 
<p align="center">
<img src="https://raw.githubusercontent.com/riveSunder/yuca/master/assets/zoo/lenia_zoo/hydrogeminium_natans_glider_00.gif">
</p>

_Hydrogeminium natans_ is a Lenia rule set with an update rule defined by $\mu = 0.26$ and $\sigma = 0.036$ and a neighborhood with three rings centered at $\mu_{k_n} = (0.0938, 0.2814, 0.4690)$, with $\sigma_{k_n} = (0.033, 0.033, 0.033)$ and weighted by $(0.5, 1.0, 0.667)$. The glider shown here has the same behavior (but different cell values) than fast/wobbly gliders found in s613 and s643 evolved CA, which share the _Hydrogeminium_ neighborhood kernel. 


[^Ca1974]: Carter, Brandon. "Large number coincidences and the anthropic principle in cosmology." Confrontation of cosmological theories with observational data. Springer, Dordrecht, 1974. 291-298.
[^Ca1984]: Carter, Brandon. "The anthropic principle and its implications for biological evolution." Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 310.1512 (1983): 347-363.
[^Sa1980]: Carl Sagan. Cosmos. Random House, New York. 1980. ISBN: 0-394-50294-9 p. 218
[^Ch2018a]:B. W.-C. Chan, "Lenia: Biology of Artificial Life," Complex Systems, 28(3), 2019 pp. 251–286. https://doi.org/10.25088/ComplexSystems.28.3.251
[^Ch2018b]: [https://chakazul.github.io/Lenia/JavaScript/Lenia.html](https://chakazul.github.io/Lenia/JavaScript/Lenia.html)

