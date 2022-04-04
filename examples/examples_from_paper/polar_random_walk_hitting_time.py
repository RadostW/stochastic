import pychastic
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax

y_drift = 4.0
sigma = 1

problem = pychastic.sde_problem.SDEProblem(
    a=lambda x: jnp.array([1 / (2 * x[0]) + y_drift*jnp.sin(x[1]), y_drift*jnp.cos(x[1]) / x[0]]),
    b=lambda x: sigma*jnp.array(
        [[jnp.cos(x[1]), jnp.sin(x[1])], [-jnp.sin(x[1]) / x[0], jnp.cos(x[1]) / x[0]]]
    ),
    x0=jnp.array([2.0, 0.0]),
    tmax=1.0,
)

solver = pychastic.sde_solver.SDESolver()
solver.scheme = 'milstein'

dt = 2**(-8)
n_samples = 100

solver.dt = dt
solution = solver.solve_many(problem,n_samples,seed=0)
x = solution['solution_values'][..., 0]*jnp.cos(solution['solution_values'][..., 1])
y = solution['solution_values'][..., 0]*jnp.sin(solution['solution_values'][..., 1])

barrier = 2
hit_occured_in_solution = (y > barrier).any(axis=1)
trajectories = jnp.stack([x, y], axis=-1)[hit_occured_in_solution]  # (n_traj, step, coord)

def process_trajectory(trajectory):
    # should be applied to trajs that crossed barrier
    hit_index = (trajectory[:, 1] > barrier).argmax()
    
    position_before = trajectory[hit_index-1]
    position_after = trajectory[hit_index]
    
    y_before = position_before[1]
    y_after = position_after[1]

    weight = (barrier - y_before)/(y_after - y_before)
    
    interpolated_hit_place = position_after*weight + position_before*(1-weight)
    interpolated_hit_time = dt*(hit_index+weight)
    
    return {
        'interpolated_hit_place': interpolated_hit_place,
        'interpolated_hit_time': interpolated_hit_time
    }

process_trajectory(trajectories[0])
data = jax.vmap(process_trajectory)(trajectories)

# hit time histogram vs theory
plt.figure()

plt.hist(np.array(data['interpolated_hit_time']), bins=min(50, n_samples//10), density=True)

t = np.linspace(0, problem.tmax, 100)[1:]
theoretical_hitting_time_desity = barrier/(sigma*np.sqrt(2*np.pi*t**3))*np.exp(-(barrier-y_drift*t)**2/(2*sigma**2*t))
plt.plot(t, theoretical_hitting_time_desity)
plt.show()
plt.savefig('stopping_time_hist.png')
plt.close()

# stopped processes

def plot_stopped_trajectories(fraction_of_time, **kwargs):    
    trimmed_trajectories = trajectories[:, :int(fraction_of_time*problem.tmax/dt)]
    hit_occured = (trimmed_trajectories[:, :, 1] > barrier).any(axis=1)
    trimmed_trajectories = trimmed_trajectories[hit_occured]
    data = np.array(
        jax.vmap(process_trajectory)(trimmed_trajectories)['interpolated_hit_place'].T
    )
    #data = np.array(trimmed_trajectories[:, -1, :].T)
    plt.scatter(*data, alpha=0.2, label=f"{fraction_of_time:.2f}", **kwargs)

plt.figure()
plot_stopped_trajectories(1/3)
plot_stopped_trajectories(1/2)
plt.legend()
plt.savefig('stopped.png')
plt.close()

# comparison of solution hit places vs wiener hit places (interpolated)