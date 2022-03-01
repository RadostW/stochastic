Advanced features of Pychastic
==============================

Memory management - sample discarding
'''''''''''''''''''''''''''''''''''''

In many cases (for example stiff equations) even though small step is required
we are not really interested in values at all steps. We can use this to our
advantage by discarding evaluation points which are not needed. You can accompish
this behaviour using `chunk_size` option of `solve` and `solve_many` methods.

.. prompt:: python >>> auto

    >>> TODO: EXAMPLE CODE

Memory management - randomization discarding
''''''''''''''''''''''''''''''''''''''''''''

It is much more efficient to compute all random values in advance and not generate
noise at each step. Sadly this comes at memory cost. By default all randomization
is done before integrating trajectories, we can split this process into smaller
runs by using `chunks_per_randomization` option of `solve` and `solve_many` methods.

.. prompt:: python >>> auto

    >>> TODO: EXAMPLE CODE

Step post-processing aka domain hacks
'''''''''''''''''''''''''''''''''''''

Whenever we are anaysing SDE or ODE on a manifold whose universal cover is not 
:math:`\mathbb{R}^n` we need to handle switching between charts. Similarily, if
we know that some constraint is satisfied exactly in our system (conservation of 
energy or momentum for example) we overcome error accumulation in conserved 
variables by projecting onto desired subspace after each step. Both of those needs
can be acomplished using `step_post_processing` option of `solve` and `solve_many`
methods. It takses a function which maps location in phase space onto adjusted new
location in phase space after each step.

.. prompt:: python >>> auto
    
    >>> TODO: EXAMPLE CODE

Multiple starting positions
'''''''''''''''''''''''''''

TODO: This feature is work in progress.

Sometimes we are interested in equilibrium properties of studied system. It is beneficial
to pick different initial conditions for each sample path in such way that starting
condition is distributed according to equilibrium configuration. Such behaviour 
is accomplished setting `x0` value in SDEProblem to a list of initial conditions 
and setting `n_trajectories` to sde_solver.automatic

.. prompt:: python >>> auto
    
    >>> TODO: EXAMPLE CODE


