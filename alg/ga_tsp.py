"""
Genetic Algorithm class for ENDOF (Endof New Distributed Optimiaztion Framework)

This is a derived class for solving a standard TSP using ga; it is meant
primarily as an example showing how to adapt the base class to a specific
problem.

Copyright Diego Diaz Fidalgo 2014/08/20

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import random
from ga import ga

class ga_tsp(ga):
    """
    Genetic Algorithm for solving TSP problems
    """
    def __init__(self, cost_matrix, pop_size=50, elitism=0, crossover_prob=0.5,
                 mutation_prob=0.5, max_iter=1000, rand_seed=None,
                 rand_offset=0):
        """
        Initialization of the GA:
        The following parametres are needed (defaults will be used if no
        value is provided, except for the number of genes)
        - The cost matrix for the problem as a list of lists where
          cost_matrix[a][b] is the cost of the arc from a to b. Symmetry is not
          assumed. The chromosome size will be deduced as len(cost_matrix)
        - Population size
        - Elitism: individuals carried over from one iteration to the next
        - Crossover Probability
        - Mutation Probability
        - Maximum number of iterations
        - Random Seed: a seed can be provided to replicate a result; if no seed
          is provided, the rng is initialized in the standard fashion
        - Random Offset: If a number > 0 is provided, the state of the rng will
          be advanced by that quantity (useful for parallel runs with
          deterministic seeds by providing a different offset for each instance)
        Additionally, the GA is reset by instantiating an empty population
        and setting the best solution and fitness value to None, and
        zeroing the iteration counter
        """
        self._cost_matrix = cost_matrix
        ng = len(cost_matrix)
        super().__init__(ng, pop_size, elitism, crossover_prob, mutation_prob,
                         max_iter, rand_seed, rand_offset)

    def initialize_population(self):
        """
        Initialize the population

        Create a number of random permutations of the cities.
        """

        cities = range(self._num_genes)
        pop = []
        for _ in range(self._pop_size):
            new_indiv = cities[:]
            self._random.shuffle(new_indiv)
            pop.append(new_indiv)
            
        self._pop = self._rank_pop(pop)

    def _crossover(self, parent1, parent2):
        """
        Crossover operation

        Using a variation of order crossover: select a chunk of the first
        parent given by two cut points, and place it in the same position in
        the offspring; then fill the rest of the offspring with the remaining
        elements in the order they appear in the second parent
        """
        # select cutting points (0 to num_genes, as there is the possibility to
        # have the chunk include the last element when slicing by idx1:idx2)
        idx1 = self._random.randint(0, self._num_genes)
        # For the second, the first one is not viable
        idx2 = self._random.randint(0, self._num_genes - 1)
        # Offset the missing potential value idx1 if needed
        if idx2 >= idx1:
            idx2 += 1
        else:  # invert the indices so that idx1 < idx2
            idx1, idx2 = idx2, idx1

        chunk = parent1[idx1:idx2]
        others = parent2[:]
        for elem in chunk:
            others.remove(elem)
        # Build offspring
        offspring = others[:idx1]
        offspring.extend(chunk)
        offspring.extend(others[idx1:])

        return offspring


    def _mutation(self, indiv):
        """
        Mutation operation

        Swap the position of two random elements.
        """
        # Select two random positions from 0 to num_genes -1 as it is a
        # position index (cf. the use in crossover)
        idx1 = self._random.randint(0, self._num_genes - 1)
        # For the second, the first one is not viable
        idx2 = self._random.randint(0, self._num_genes - 2)
        # Offset the missing potential value idx1 if needed
        if idx2 >= idx1:
            idx2 += 1
        new_indiv = indiv[:]
        new_indiv[idx1], new_indiv[idx2] = new_indiv[idx2], new_indiv[idx1]
        return new_indiv


    def _fitness(self, indiv, close_tour=True):
        """
        Fitness function

        The sum of the costs of the selected arcs for the tour as taken from
        the cost matrix.

        Close tour means that the cost of the circular tour is considered,
        counting the arc from the last to the first element of the sequence.
        Otherwise, this arc is not included.
        """
        shifted_indiv = indiv[1:]
        # Add the first element if closing the tour for the zip to include
        # the last to first elements arc. As zip is truncated to the shortest
        # sequence, just not adding this is enough when not colsing the loop
        if close_tour:
            shifted_indiv.append(indiv[0])
        arcs = zip(indiv, shifted_indiv)
        return sum(self._cost_matrix[o][d] for (o, d) in arcs)


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
    myga = ga_tsp(cm, elitism=2)
    myga.initialize_population()
    myga._run()
    print(myga._best_sol, myga._best_obj)
