
## Description
This page corresponds to the article on my blog about the <a href='https://yenhsunlin.github.io/2021/06/18/pandemic/'>pandemic spreading</a> .

## Prerequisites

- `numpy`
- `scipy`
- `cv2`

If python OpenCV (`cv2`) is not installed, the script cannot be run properly. Please do the following on the prompt

> `conda install opencv`

or

> `pip install opencv`

## Known issue

Multiprocessing drawing function `drawsim_mp` is relatively efficient in drawing figures with small size simulation data, compared with its cousin `drawsim`. For size larger than 100 MB, please avoid using this function since it will not improve the speed but causes crashing issue.

This is due to `multiprocessing.Pool` map the simulation data with the subroutine `_drawfullout` to each CPU core, for simulation file size is too large, it will consume huge amount of memory and causes the system unstable even crashed. Lowering the number of CPU core used in `drawsim_mp` will not fix this problem. 

This could be resolved in the next revision.

## Change log

- 2021-06-20: Adding multiprocessing support for drawing process, the associated function is `drawsim_mp`
