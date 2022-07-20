# ... We can implement Morley by splitting the growth function

## Instead of a single growth function, we use two functions: genesis and persistence, conditioned on the current cell value

{:style="text-align:center;"}
![glaberish equation](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/equations/glaberish.png)

## This allows non-overlapping updates

{:style="text-align:center;"}
![Morley genesis function in glaberish](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/morley_in_glaberish.png)

## With growth split into genesis and persistence, glaberish recovers the ability to implement Morley and others. 

{:style="text-align:center;"}
![Morley puffer in Glaberish (works)](https://raw.githubusercontent.com/riveSunder/yuca/master/assets/glaberish/morley_puffer_glaberish.gif)

{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/yuca/g_slide_006) -- [Next slide](https://rivesunder.github.io/yuca/g_slide_008)
