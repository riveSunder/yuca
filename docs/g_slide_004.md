# Life in Lenia

## Cellular Automata (CA)

### In general CA are dynamic systems that change due to rules applied to local context.

### _i.e._ compute neighborhood > compute update

{:style="text-align:center;"}
![Moore Neighborhood](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/moore_neighborhood.png)
![Morley glider](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/morley_glider_000.png)

## Lenia

### In Lenia, the neighborhood is defined by a smooth, continuously valued convolution kernel with a transition update defined by a smooth function call a Growth function (often one or more Gaussians)

![Lenia standard Orbium rule](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/lenia_orbium.png)

### The update takes the essential form of Euler's numerical method for differential equations.  

{:style="text-align:center;"}
![Lenia equation](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/lenia.png)

## Life in Lenia

### In words, the *B3/S23* Life rule: if a cell has exactly 3 neighbors it becomes a 1, if it has 2 or 3 neighbors it keeps its current value, and if it has any other number of neighbors it becomes 0. 

### The Lenia framework can easily represent Conway's Game of Life with a Moore neighborhood convoluton kernel and smooth step functions corresponding to B3/S23.

{:style="text-align:center;"}
![Lenia standard Orbium rule](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/life_in_lenia.png)


{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/yuca/g_slide_003) -- [Next slide](https://rivesunder.github.io/yuca/g_slide_005)
