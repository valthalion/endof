"""
Ant Colony Optimization class for ENDOF (Endof New Distributed Optimiaztion
Framework)

This is a derived class for solving a standard TSP using aco; it is meant
primarily as an example showing how to adapt the base class to a specific
problem.

Copyright Diego Diaz Fidalgo 2014/08/23

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import random
from aco import aco
import operator


class aco_tsp(aco):
    """
    ACO for solving the travelling salesman problem
    """
    def __init__(self, cost_matrix, num_ants=50, default_ph=1,
                 evaporation=0.95, heuristics=None, num_ants_ph=3, elitism=1,
                 alpha=1, beta=1, max_iter=1000, rand_seed=None, rand_offset=0):
        """
        Initializaton of the ACO:
        The following parametres are needed (defaults will be used if no value
        is provided, except for the cost matrix)
        - Cost matrix as a list of lists where cost_matrix[a][b] is the cost of
          the arc going from a to b. cost_matrix[x][x] are not used, but needed
          as the solution length is calculated as len(cost_matrix). Costs are
          not assumed to be symmetric
        - Number of ants per iteration
        - Default pheromone level for matrix initialization
        - Evaporation: the factor to appl for pheromone evaporation (e.g. the
          0.8 default means 20% evaporation)
        - Heuristics: the values to use as heuristics when determining
          candidates. Make sure to use the format that you expect in your
          ant() method. The default is None, in which case the arc cost is used
          as heuristic. If provided, this argument overrides that and
          heuristics[a][b] is used for element b following element a in the tour
        - Number of ants in each iteration that deposit pheromone
        - Elitism: the number of best solutions so far that deposit pheromone
          each iteration (apart from the iteration ants)
        - alpha, beta: parametres of the pheromone and heuristic evaluation for
          candidate selection (see seminal paper by Dorigo)
        - Maximum number of iterations
        - Random Seed: a seed can be provided to replicate a result; if no seed
          is provided, the rng is initialized in the standard fashion
        - Random Offset: If a number > 0 is provided, the state of the rng will
          be advanced by that quantity (useful for parallel runs with
          deterministic seeds by providing a different offset for each instance)
        Additionally, the ACO is reset by instantiating an empty population
        and setting the best solution and fitness value to None, zeroing the
        iteration counter and resetting the pheromone matrix
        """

        # TODO: Find good default values for params
        self._cost_matrix = cost_matrix
        sol_len = len(cost_matrix)
        # Default heuristic is 1 / (1 + arc cost), but use alternate if provided
        if heuristics is not None:
            heur = heuristics
        else:
            heur = [[1.0 / (1 + x) for x in cost_row]
                    for cost_row in cost_matrix]
#        print(cost_matrix)
#        print(heur)
#        print()
        super().__init__(sol_len, num_ants, default_ph, evaporation, heur,
                         num_ants_ph, elitism, alpha, beta, max_iter,
                         rand_seed, rand_offset)

    def init_ph(self):
        """
        Initialize the pheromone matrix

        The pheromone matrix is an n-by-n matrix where ph[a][b] is the pheromone
        associated to element b following element a in the tour
        """
        sl = self._sol_length
        self._pheromones = [[self._default_ph] * sl for _ in range(sl)]

    def ant(self):
        """
        Behaviour of each ant

        Build a tour by selecting each step according to heuristics and
        pheromone levels in the corresponding arcs. Since tours are closed, the
        selection of the first element can be arbitrary, and we choose element
        zero. Keep adding elements until all are used (full tour)
        """
        mysol = []
        cities = list(range(self._sol_length))
        mysol.append(cities.pop(0))
        while cities:
            # All candidate arcs start at the last city in the tour so far
            orig = mysol[-1]
            # Get the pheromones and heuristics for arcs from orig to the
            # remaining candidate cities
            pheroms = (self._pheromones[orig][s] for s in cities)
            heurs = (self._heuristics[orig][s] for s in cities)
            # Same origin and destination is not possible because origin has
            # already been popped out of cities, therefore no filtering in
            # pheroms and heurs above
            probs = [self._pow_alpha(pheromone) * self._pow_beta(heur)
                     for pheromone, heur in zip(pheroms, heurs)]
            sum_probs = sum(probs)
            rnd_num = self._random.uniform(0, sum_probs)
            select = -1
            cum_prob = 0
            while cum_prob <= rnd_num:  # Always goes in at least once
                select += 1
                if select > len(cities):
                    print(select)
                    print(cities)
                    print(rnd_num)
                    print(cum_prob)
                    print(probs)
                    print(sum_probs)
                    print(self._pheromones)
                    print(self._heuristics)
                cum_prob += probs[select]
            mysol.append(cities.pop(select))
        return mysol

    def _fitness(self, indiv):
        """
        Fitness function

        Calculated as the sum of the costs associated to the selected arcs for
        the tour, taken from the cost matrix. The tour is cosidered closed,
        i.e. the cost from the last to the first element is included.
        """
        shifted_indiv = indiv[1:] + [indiv[0]]
        arcs = zip(indiv, shifted_indiv)
        cost = sum(self._cost_matrix[o][d] for (o,d) in arcs)
        return cost

    def _update_ph(self, pop):
        """
        Update the pheromone matrix for each (sol, fitness) tuple in population

        The update is by 1 /(1 + fitness) to avoid problems with division by
        zero.
        """
        for sol, fit in pop[:self._num_ants_ph]:
            ph = 1.0 / (1 + fit)
            for s in range(self._sol_length-1):
                self._pheromones[sol[s]][sol[s+1]] += ph
            self._pheromones[sol[self._sol_length-1]][sol[0]] += ph

    def _evaporate(self):
        """
        Apply evaporation
        """

        for s in range(self._sol_length):
            for p in range(self._sol_length):
                self._pheromones[s][p] *= self._evaporation


if __name__ == "__main__":
    cm = [[0, 4, 4, 4, 4, 4, 4, 4, 4, 1],
          [1, 0, 4, 4, 4, 4, 4, 4, 4, 4],
          [4, 1, 0, 4, 4, 4, 4, 4, 4, 4],
          [4, 4, 1, 0, 4, 4, 4, 4, 4, 4],
          [4, 4, 4, 1, 0, 4, 4, 4, 4, 4],
          [4, 4, 4, 4, 1, 0, 4, 4, 4, 4],
          [4, 4, 4, 4, 4, 1, 0, 4, 4, 4],
          [4, 4, 4, 4, 4, 4, 1, 0, 4, 4],
          [4, 4, 4, 4, 4, 4, 4, 1, 0, 4],
          [4, 4, 4, 4, 4, 4, 4, 4, 1, 0],]
    max_it = 100
    myaco = aco_tsp(cm, max_iter=max_it)
    myaco._run()
    print(myaco._best_sol, myaco._best_obj)
    print(myaco._pheromones)
