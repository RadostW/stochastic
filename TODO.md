todo
- code
  - speed optimization for adaptive methods
  - adaptive?
  - docs

- paper
  - neat tables with schemes, etc.
  - literature review
  - speed comparison
  - langevin
  - briefly explain methods
  - hydrodynamics brownian motion equations in ito form
  - examples of use
  - trajectories (ensembled)
  - error vs time, error vs steps (adaptive case)
  - hardware acceleration

- package
  - [ ] readme


Notes
```
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
```
might help with `OpenBLAS blas_thread_init` error.

to debug nans set envvar `JAX_DEBUG_NANS=True` or `from jax.config import config; config.update("jax_debug_nans", True)`
