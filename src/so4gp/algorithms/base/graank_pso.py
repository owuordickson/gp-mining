# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import time
import random
import numpy as np
from .graank_base import BaseGrad


class ParticleGRAANK(BaseGrad):

    def __init__(self, *args, max_iter: int = 1, n_particle: int = 5, vel: float = 0.9,
                 coeff_p: float = 0.01, coeff_g: float = 0.9, **kwargs):
        """
        Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI), and its
        quality is measured by its computed support value. For example, given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
            position and fitness. The fitness is derived from the computed support of that candidate, the higher the
            support value, the higher the fitness. The aim of the algorithm is to search through a population of particles
            (or candidates) and find those with the highest fitness as efficiently as possible.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :param n_particle: [optional] initial particle population, default is 5
        :param vel: [optional] velocity, default is 0.9
        :param coeff_p: [optional] personal coefficient, default is 0.01
        :param coeff_g: [optional] global coefficient, default is 0.9

        """
        super(ParticleGRAANK, self).__init__(*args, **kwargs)
        self._max_iteration: int = max_iter
        self._n_particles: int = n_particle
        self._velocity: float = vel
        self._coeff_p: float = coeff_p
        self._coeff_g: float = coeff_g

    def discover(self, target_col: int | None = None, time_data: dict|None= None, exclude_target: bool = False) -> dict:
        """
        Searches through particle positions to find GP candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :param target_col: Target feature's column index.
        :param time_data: (optional) time data for estimating time lag.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.

        :return: A dict object
        """

        start = time.time()
        self.target_col = target_col
        try:
            self.init_search_space(self._n_particles)
            s_space = self.search_space
            if s_space is None:
                return {"Error": "Search space is empty!"}
        except ValueError as e:
            return {"Error": e}

        pbest_pop = s_space.pop.copy()
        gbest_particle = pbest_pop[0]
        velocity_vector = np.ones(self._n_particles)
        while s_space.iter_count < self._max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(self._n_particles):
                self.evaluate_candidate(s_space.pop[i], exclude_target, time_data=time_data)
                part_cost = s_space.pop[i].cost
                part_pos = s_space.pop[i].position
                if part_cost is not None and pbest_pop[i].cost is not None and gbest_particle.cost is not None:
                    if pbest_pop[i].cost > part_cost:
                        pbest_pop[i].cost = part_cost
                        pbest_pop[i].position = part_pos

                    if gbest_particle.cost > part_cost:
                        gbest_particle.cost = part_cost
                        gbest_particle.position = part_pos
            # if abs(gbest_fitness_value - self.target) < self.target_error:
            #    break
            if gbest_particle.cost is not None and s_space.best_candidate.cost is not None:
                if s_space.best_candidate.cost > gbest_particle.cost:
                    s_space.best_candidate = BaseGrad.Candidate(position=gbest_particle.position, cost=gbest_particle.cost)

            for i in range(self._n_particles):
                part_pos = s_space.pop[i].position
                if part_pos is not None and pbest_pop[i].position is not None and gbest_particle.position is not None:
                    new_velocity = (self._velocity * velocity_vector[i]) + \
                               (self._coeff_p * random.random()) * (pbest_pop[i].position - part_pos) + \
                               (self._coeff_g * random.random()) * (gbest_particle.position - part_pos)
                    s_space.pop[i].position = s_space.pop[i].position + new_velocity

            # Increment iteration count
            s_space.iter_count += 1

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "PSO-GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Initial Population": f"{self._n_particles}",
            "Velocity": f"{self._velocity}",
            "Personal coefficient": f"{self._coeff_p}",
            "Global coefficient": f"{self._coeff_g}",
            "Number of iterations": f"{s_space.iter_count}",
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{s_space.invalid_count}"}
        return out_dict
