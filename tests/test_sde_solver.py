import jax
import pytest
from pychastic.sde_solver import SDESolver
from pychastic.sde_problem import SDEProblem
import numpy as np

a = 1
b = 1
scalar_geometric_bm = SDEProblem(
    a=lambda x: a * x,
    b=lambda x: b * x,
    x0=1.0,
    tmax=1.0,
    exact_solution=lambda x0, t, w: x0 * np.exp((a - 0.5 * b * b) * t + b * w),
)

a = 1.0
scalar_arctan_problem = SDEProblem(
    a=lambda x: -(a ** 2) * jax.numpy.sin(x) * jax.numpy.cos(x) ** 3,
    b=lambda x: a * jax.numpy.cos(x) ** 2,
    x0=0.0,
    tmax=1.0,
    exact_solution=lambda x0, t, w: np.arctan(a * w + np.tan(x0)),
)


@pytest.mark.parametrize(
    "solver,problem,steps,quantile_99",
    [
        (SDESolver(), scalar_geometric_bm, 2 ** 7, 1.4),
        (SDESolver(scheme="milstein"), scalar_geometric_bm, 2 ** 7, 0.7),
        (SDESolver(), scalar_arctan_problem, 2 ** 7, 0.09),
        (SDESolver(scheme="milstein"), scalar_arctan_problem, 2 ** 7, 0.008),
    ],
)
def test_again_exact_solution_scalar(solver: SDESolver, problem, steps, quantile_99):
    solver.dt = problem.tmax / steps

    result = solver.solve(problem)
    time_values = result['time_values'].reshape(-1, 1)
    solution_values = result["solution_values"]
    wiener_values = result["wiener_values"]

    exact_values = problem.exact_solution(problem.x0, time_values, wiener_values)
    errors = exact_values - solution_values
    end_error = errors[-1]
    assert abs(end_error) < quantile_99  # .99 quantile


polar_random_walk_problem = SDEProblem(
    a=lambda x: jax.numpy.array([1 / (2 * x[0]), 0]),  # [1/2r,0]
    b=lambda x: jax.numpy.array(
        [
            [jax.numpy.cos(x[1]), jax.numpy.sin(x[1])],  # cos \phi,      sin \phi
            [
                -jax.numpy.sin(x[1]) / x[0],
                jax.numpy.cos(x[1]) / x[0],
            ],  # -sin \phi / r, cos \phi / r
        ]
    ),
    x0=jax.numpy.array([1.0, 0.0]),  # r=1.0, \phi=0.0
    tmax=1.0,
)
polar_random_walk_problem.to_cartesian = lambda x: jax.numpy.array(
    [x[0] * jax.numpy.cos(x[1]), x[0] * jax.numpy.sin(x[1])]
)


parabolic_random_walk_problem = SDEProblem(
    lambda x: jax.numpy.array(
        [
            (x[0] - x[1]) / (2 * (x[0] + x[1]) ** 2),
            (-x[0] + x[1]) / (2 * (x[0] + x[1]) ** 2),
        ]
    ),
    lambda x: jax.numpy.array(
        [
            [1 / (2 * x[0] + 2 * x[1]), x[1] / (x[0] + x[1])],
            [-1 / (2 * x[0] + 2 * x[1]), x[0] / (x[0] + x[1])],
        ]
    ),
    x0=jax.numpy.array([1.0, 0.0]),  # u=1 , v=0
    tmax=0.05,
)

parabolic_random_walk_problem.to_cartesian = lambda x: jax.numpy.array(
    [x[0] ** 2 - x[1] ** 2, x[0] + x[1]]
)


@pytest.mark.parametrize(
    "solver,problem,steps,quantile_99",
    [
        (SDESolver(), polar_random_walk_problem, 2 ** 7, 1.08),
        (SDESolver(scheme="milstein"), polar_random_walk_problem, 2 ** 7, 1.2),
        (SDESolver(), parabolic_random_walk_problem, 2 ** 7, 0.025),
        (SDESolver(scheme="milstein"), parabolic_random_walk_problem, 2 ** 7, 0.002),
    ],
)
def test_again_exact_solution_vector(solver: SDESolver, problem, steps, quantile_99):
    solver.dt = problem.tmax / steps

    result = solver.solve(problem, seed=1)
    # time_values = result["time_values"].reshape(-1, 1)
    solution_values = result["solution_values"]
    wiener_values = result["wiener_values"]

    end_error = problem.to_cartesian(solution_values[-1]) - (
        wiener_values[-1] + problem.to_cartesian(problem.x0)
    )
    l2_end_error = (end_error ** 2).sum() ** 0.5
    assert abs(l2_end_error) < quantile_99  # .99 quantile


def test_solve_many_handles_multiple_initial_conditions():
    solver = SDESolver(scheme="milstein", dt=1e-2)
    problem = scalar_geometric_bm
    n_copies = 4
    problem.x0 = jax.numpy.tile(problem.x0, (n_copies, 1))

    with pytest.raises(ValueError):
        solver.solve_many(problem, n_trajectories=1)

    result = solver.solve_many(problem, n_trajectories=None)
    
    assert result['solution_values'].ndim == 3
    assert result['solution_values'].shape[0] == n_copies
