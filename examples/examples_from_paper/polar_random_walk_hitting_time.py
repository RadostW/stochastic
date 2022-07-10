import pychastic
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

y_drift = 4.0
sigma = 1
y_barrier = 2

def drift(x):
    return jnp.array(
        [1 / (2 * x[0]) + y_drift * jnp.sin(x[1]), y_drift * jnp.cos(x[1]) / x[0]]
    )

def noise(x):
    return sigma * jnp.array(
        [[jnp.cos(x[1]), jnp.sin(x[1])], [-jnp.sin(x[1]) / x[0], jnp.cos(x[1]) / x[0]]]
    )

problem = pychastic.sde_problem.SDEProblem(
    a= drift,
    b= noise,
    x0=jnp.array([2.0, 0.0]),
    tmax=2.0,
)

n_samples = 10000

solver = pychastic.sde_solver.SDESolver()
solver.scheme = "euler"
solver.dt = 2 ** (-10)

def process_trajectory(trajectory):
    hit_index = (trajectory[:, 1] > y_barrier).argmax()

    position_before = trajectory[hit_index - 1]
    position_after = trajectory[hit_index]

    y_before = position_before[1]
    y_after = position_after[1]

    # linear interpolation
    weight = (y_barrier - y_before) / (y_after - y_before)
    interpolated_hit_place = position_after * weight + position_before * (1 - weight)
    interpolated_hit_index = hit_index + weight

    return {
        "interpolated_hit_place_x": interpolated_hit_place[0],
        "interpolated_hit_place_y": interpolated_hit_place[1],
        "interpolated_hit_index": interpolated_hit_index,
    }

process_trajectory_vmapped = jax.vmap(process_trajectory)


def rmse(tensor):
    return jnp.sqrt((tensor ** 2).mean())


def analyze_solver(solver, plots=False):
    solution = solver.solve_many(problem, n_samples, seed=0, chunks_per_randomization = 3)

    w1 = solution["wiener_values"][..., 0]
    w2 = solution["wiener_values"][..., 1]
    x_real = w1 + 2
    y_real = w2 + y_drift * solution["time_values"][0]

    r_sol = solution["solution_values"][..., 0]
    phi_sol = solution["solution_values"][..., 1]
    x_sol = r_sol * jnp.cos(phi_sol)
    y_sol = r_sol * jnp.sin(phi_sol)

    hit_occured_in_solution = (y_sol > y_barrier).any(axis=1)
    hit_occured_in_real = (y_real > y_barrier).any(axis=1)
    hit_occured_in_both = hit_occured_in_solution & hit_occured_in_real

    # (n_traj, step, coord)
    trajectories_sol = jnp.stack([x_sol, y_sol], axis=-1)[hit_occured_in_both]
    trajectories_real = jnp.stack([x_real, y_real], axis=-1)[hit_occured_in_both]

    processed_sol = process_trajectory_vmapped(trajectories_sol)
    processed_real = process_trajectory_vmapped(trajectories_real)

    hit_time_sol = processed_sol["interpolated_hit_index"] * solver.dt
    hit_time_real = processed_real["interpolated_hit_index"] * solver.dt

    hit_time_error = hit_time_sol - hit_time_real

    hit_time_rmse = rmse(hit_time_error).item()
    hit_time_error_std = hit_time_error.std().item()

    hit_time_mean = jnp.mean(hit_time_sol)
    hit_time_mean_std = jnp.std(hit_time_sol) / jnp.sqrt(len(hit_time_sol))

    return {
        "hit_time_sol": hit_time_sol,
        "hit_time_real": hit_time_real,
        "hit_time_rmse": hit_time_rmse,
        "hit_time_error_std": hit_time_error_std,
        "hit_time_mean": hit_time_mean,
        "hit_time_mean_std" : hit_time_mean_std,
    }


#schemes = ['euler', 'milstein', 'wagner_platen']
schemes = ["wagner_platen"]
n_steps_list = [int(x) for x in 1.5 ** np.arange(3, 20)]
results = []

disp = dict()

for scheme in schemes:
    for n_steps in n_steps_list:
        print(scheme, n_steps)
        dt = problem.tmax / n_steps
        solver = pychastic.sde_solver.SDESolver(dt=dt, scheme=scheme)
        result = analyze_solver(solver)
        print(f"{result['hit_time_mean']-0.5=}")
        print(f"+-{result['hit_time_mean_std']=}")
        #print(f"{result['hit_place_x_mean']=}")
        data = {
            k: v
            for k, v in result.items()
            if "error" in k or "rmse" in k or "mean" in k
        }
        data["scheme"] = scheme
        data["n_steps"] = n_steps
        results.append(data)
        disp[n_steps] = result['hit_time_mean']
    [print("{" + f"{k},{v}" +"}") for k,v in disp.items()]


