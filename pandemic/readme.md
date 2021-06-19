
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

`drawsim_mp` is relatively efficient in drawing figures with small size simulation data. For size larger than 100 MB, please avoid using this function since it will not improve the speed but causes crashing issue.

This is due to `multiprocessing.Pool` map the simulation data with the subroutine `_drawfullout` to each CPU core, for simulation file size is too large, it will consume hugh amount of memory and cause the system unstable even crashed. Lowering the number of CPU core used in `drawsim_mp` will not fix this problem.

## Change log

- 2021-06-20: Adding multiprocessing support for drawing process, the associated function is `drawsim_mp`
