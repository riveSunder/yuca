# How do typical glaberish and Lenia CA compare?

## One apparent difference is that most Lenia rules from Chan's taxonomy [^Ch2019] (Interactive online zoo [here](https://chakazul.github.io/Lenia/JavaScript/Lenia.html)) form Turing patterns from random uniform initializations. 

{:style="text-align:center;"}
![teaser figure showing Orbium and s613 CA](https://raw.githubusercontent.com/riveSunder/yuca_docs/master/assets/glaberish/typical_cca.gif)

### Examples of continuous CA: a) CA evolved with random fitness b) Lenia CA from Chan's taxonomy [^Ch2019] c) CA in glaberish evolved for roughly even chances of random initial grids vanishing d) CA in glaberish evolved by selection for vanishing unpredictability [^Da2022] 
{:style="text-align:center;"}

## What about capturing the differences between CA dynamics with an information-theoretic measure [^note]

1. Convert grid to 8-bit integer format
2. For each neighborhood-sized window in the grid:

    * Compute relative proporitions of each of 256 cell values 
    * Compute entropy for each cell location according to 

{:style="text-align:center;"}
![entropy equation](https://raw.githubusercontent.com/riveSunder/yuca_docs/master/assets/equations/spatial_entropy_annotated.png)

[^Ch2019]: Chan, Bert Wang-Chak. "Lenia - Biology of Artificial Life." Complex Syst. 28 (2019): [https://arxiv.org/abs/1812.05433](https://arxiv.org/abs/1812.05433).

[^Da2022]: Davis, Q. Tyrell, and Josh Bongard. "Selecting Continuous Life-Like Cellular Automata for Halting Unpredictability: Evolving for Abiogenesis." GECCO Companion Proceedings (2022). [https://arxiv.org/abs/2205.10463](https://arxiv.org/abs/2205.10463)

[^note]: Entropy and other information theoretic measures abound for making sense of CA complexity. See the background and references in the [paper](https://arxiv.org/abs/2205.10463) for more examples.



{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/yuca_docs/g_slide_007) -- [Next slide](https://rivesunder.github.io/yuca_docs/g_slide_009)
