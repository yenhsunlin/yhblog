
## Description

This page corresponds to the article (in Chinese) on my blog about the <a href='https://yenhsunlin.github.io/2021/06/18/pandemic/'>*pandemic spreading*</a>. The goal of this simulation is to simulate how a disease evolving with time with given conditions, eg. infection, recovery and death. With and without mask wearing is also quantified in the simulation.

The method adopted in this project can be considered as non-interacting *N*-body simulation with MCMC probabilistic sampling. While non-interacting means no collisions between subjects when they encounter each other. Infection condition only depends on how close the healthy subject with the ill subject and the duration of the healthy subject stays in the infectious zone. See the article for more detail.


### Data saving with various IDEs

- *Jupyter*: commonly under the same directory as the notebook
- *Prompt*: commonly under `~/`, depends on the OS setting

## Prerequisites

- `numpy`
- `scipy`
- `cv2`

If python OpenCV (`cv2`) is not installed, the script cannot be run properly. Please do the following on the prompt

> `conda install --channel https://conda.anaconda.org/menpo opencv3`

or

> `pip install opencv-python`

## Known issue

## Past issue

Multiprocessing drawing function `drawsim_mp` is relatively efficient in drawing figures with small size simulation data, compared with its cousin `drawsim`. For size larger than 100 MB, please avoid using this function since it will not improve the speed but causes crashing issue.

This is due to `multiprocessing.Pool` map the simulation data with the subroutine `_drawfullout` to each CPU core, for simulation file size is too large, it will consume huge amount of memory and causes the system unstable even crashed. Lowering the number of CPU core used in `drawsim_mp` will not fix this problem. 

This could be resolved in the next revision.

## Change log

- 2021-06-20: Memory issue of multiprocessing support for drawing process fixed
- 2021-06-20: Adding multiprocessing support for drawing process, the associated function is `drawsim_mp`
