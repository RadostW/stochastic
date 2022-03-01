import pychastic
import numpy as np


def test_kp_exercise_9_3_3():
    a = 1.5
    b = 0.1
    problem = pychastic.sde_problem.SDEProblem(
        lambda x: a * x,
        lambda x: b * x,
        x0=1.0,
        tmax=1,
        exact_solution=lambda x0, t, w: x0 * np.exp((a - b ** 2 / 2) * t + b * w),
    )

    solver = pychastic.sde_solver.SDESolver()

    dts = [2 ** -4, 2 ** -5, 2 ** -6, 2 ** -7]
    n_rows = 10
    n_wieners_per_cell = 25

    errors = []

    for dt in dts:
        solver.dt = dt
        solution = solver.solve_many(
            problem, n_trajectories=n_rows * n_wieners_per_cell
        )
        last_values = solution["solution_values"][:, -1]
        last_wieners = solution["wiener_values"][:, -1]
        last_times = solution["time_values"][:, -1]

        exact_solution_args = zip(last_values, last_times, last_wieners)

        # exact_solution(x0, t, w)
        tmp_exact = lambda x: problem.exact_solution(problem.x0, x[1], x[2])
        exact_values = [tmp_exact(x) for x in exact_solution_args]

        error = np.fabs(np.array(exact_values) - np.array(last_values))

        errors.append(error.reshape((n_rows, n_wieners_per_cell)))

    errors = np.array(errors)

    mean_error = np.mean(errors, axis=-1)
    mean_mean_error = np.mean(mean_error, axis=-1)
    std_mean_error = (2.0 / np.sqrt(n_rows)) * np.std(mean_error, axis=-1)

    rtol = 0.15
    assert np.isclose(mean_mean_error[0], 0.280, rtol=rtol)
    assert np.isclose(mean_mean_error[1], 0.140, rtol=rtol)
    assert np.isclose(mean_mean_error[2], 0.076, rtol=rtol)
    assert np.isclose(mean_mean_error[3], 0.039, rtol=rtol)

    for (dt, mu, sigma) in zip(dts, mean_mean_error, std_mean_error):

        print(f"{dt:8.3} : {mu:5.3f}+-{sigma:5.3f}")


if __name__ == "__main__":
    test_kp_exercise_9_3_3()
