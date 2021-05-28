todo
- results
  - comparison of 1 dim euler/milstein fixed/adaptive
- code
  - higher order integrals in wiener instance
  - tests for subsampling
  - configure testing framework
  - configure debugging framework
  - interface for sampling multiple trajectories
  - speed optimization
- paper
  - neat tables with schemes, coef. functions, integral covariances, optimal dt formulas
  - comparison of 1 dim euler/milstein fixed/adaptive
- package
  - configure `setup.py`
  - decide on name
  - readme

Notes
```
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
```
might help with `OpenBLAS blas_thread_init` error.