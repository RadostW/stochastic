# This file is designed to be run by pytest-benchmark as testing suite

from os import environ
import jax
import pytest
from pychastic.sde_solver import SDESolver
from pychastic.sde_problem import SDEProblem
import numpy as np

a = 1
b = 1
scalar_tanh = SDEProblem(
  a = lambda x: -x*(1-x**2),
  b = lambda x: 1-x**2,
  x0 = 0.5,
  tmax = 5.0,
  exact_solution = lambda x0, t, w: np.tanh(w + np.arctanh(x0))
)

def test_single_trajectory_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100

  solver.dt = problem.tmax / steps

  def to_benchmark():
    	return solver.solve(problem)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0

def test_2k_trajectories_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 2000 # n trajectories

  solver.dt = problem.tmax / steps

  def to_benchmark():
    return solver.solve_many(problem,n)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0

def test_2k_trajectories_milstein_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 2000 # n trajectories

  solver.dt = problem.tmax / steps
  solver.scheme = 'milstein'

  def to_benchmark():
    return solver.solve_many(problem,n)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0


def test_2k_trajectories_wagner_platen_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 2000 # n trajectories

  solver.dt = problem.tmax / steps
  solver.scheme = 'wagner-platen'

  def to_benchmark():
    return solver.solve_many(problem,n)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0


def test_20k_trajectories_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 20000 # n trajectories

  solver.dt = problem.tmax / steps

  def to_benchmark():
    return solver.solve_many(problem,n,chunk_size = 100)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0

def test_200k_trajectories_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 200000 # n trajectories

  solver.dt = problem.tmax / steps

  def to_benchmark():
    return solver.solve_many(problem,n,chunk_size = 100)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0
  
def test_2M_trajectories_speed(benchmark):

  problem = scalar_tanh
  solver = SDESolver()
  steps = 100
  n = 2000000 # n trajectories

  solver.dt = problem.tmax / steps

  def to_benchmark():
    return solver.solve_many(problem,n,chunk_size = 20,chunks_per_randomization = 1)["solution_values"].block_until_ready()
  result = benchmark(to_benchmark)

  return 0
