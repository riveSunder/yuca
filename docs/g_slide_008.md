# How do typical Glaberish and Lenia CA compare?

## One apparent difference is that most Lenia rules from Chan's taxonomy (Interactive zoo from [^Ch2019] [here](https://chakazul.github.io/Lenia/JavaScript/Lenia.html)) form Turing patterns from random uniform initializations. 

{:style="text-align:center;"}
![teaser figure showing Orbium and s613 CA](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/representative_cca.gif)

## Examples of continuous CA: a) CA evolved with random fitness b) Lenia CA from Chan's taxonomy [^Ch2019] c) CA in glaberish evolved for roughly even chances of random initial grids vanishing d) CA in glaberish evolved by selection for vanishing unpredictability [^Da2022] 

## What about capturing the differences between CA dynamics with an information-theoretic measure

1. Convert grid to uint8
2. For each neighborhood-sized window in the grid:
    a. Compute relative proporitions of each of 256 cell values 
    b. Compute entropy for each cell location according to 

{:style="text-align:center;"}
![teaser figure showing Orbium and s613 CA](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/spatial_entropy.png)

[^Ch2019]: Chan, Bert Wang-Chak. "Lenia - Biology of Artificial Life." Complex Syst. 28 (2019): [https://arxiv.org/abs/1812.05433](https://arxiv.org/abs/1812.05433).

[^Da2022]: Davis, Q. Tyrell, and Josh Bongard. "Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis." GECCO Companion Proceedings (2022). [https://arxiv.org/abs/2205.10463](https://arxiv.org/abs/2205.10463)

{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/yuca/g_slide_007) -- [Next slide](https://rivesunder.github.io/yuca/g_slide_009)