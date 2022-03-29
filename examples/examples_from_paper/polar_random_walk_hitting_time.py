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
n_samples = 10000

solver.dt = dt
solution = solver.solve_many(problem,n_samples,seed=0)
x = solution['solution_values'][..., 0]*jnp.cos(solution['solution_values'][..., 1])
y = solution['solution_values'][..., 0]*jnp.sin(solution['solution_values'][..., 1])

barrier = 2
hit_occured = (y > barrier).any(axis=1)
hit_index = (y > barrier).argmax(axis=1)  # TODO: consider interpolating exact hitting time

trajs = jnp.stack([x, y], axis=-1)  # (n_traj, step, coord)
def stop_traj(traj):
    hit_index = (traj[:, 1] > barrier).argmax()
    return jax.lax.cond(hit_index != 0, lambda _: traj[hit_index], lambda _: traj[-1], 0)

stopped_trajs = jax.vmap(stop_traj)(trajs)

plt.figure()
plt.hist(np.array(hit_index[hit_occured])*dt, bins=50, density=True)

t = np.linspace(0, problem.tmax, 100)[1:]
theoretical_hitting_time_desity = barrier/(sigma*np.sqrt(2*np.pi*t**3))*np.exp(-(barrier-y_drift*t)**2/(2*sigma**2*t))
plt.plot(t, theoretical_hitting_time_desity)
plt.savefig('stopping_time_hist.png')
plt.close()

plt.figure()
plt.scatter(*jax.vmap(stop_traj)(trajs[:, :int(problem.tmax/dt*(1/3))]).T, alpha=0.1, label='1/3')
plt.scatter(*jax.vmap(stop_traj)(trajs[:, :int(problem.tmax/dt*(1/2))]).T, alpha=0.1, label='1/2')
plt.legend()
plt.savefig('stopped.png')
plt.close()
