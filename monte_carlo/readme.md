## Description

This page corresponds to the article (in Chinese) on my blog about the *Monte Carlo method*. The notebook should provide enough explaination on the usage. I provided GPU version of the script named `ising_gpu.py`. To run this properly, an nvidia GPU card (with CUDA cores) is required. If you do want to run the GPU version, please uncomment the associated lines in the jupyter notebook and comment the CPU related lines (both cannot co-exist).

Note that when using GPU version `spinMH`, the outputs are cupy array instead of numpy array. To be compatible with other package such as `matplotlib`, you can convert them into numpy array by `cupy.asnumpy(arr)`.

## Prerequisites

- `numpy` (both)
- `scipy` (CPU)
- `cupy-cuda111` (GPU)

## Tested enviroment

- Python 3.8.5
- numpy 1.19.2
- scipy 1.5.2
- cupy-cuda111 8.5.0
- matplotlib 3.3.2
