"""
Ant Colony Optimization class for ENDOF (Endof New Distributed Optimiaztion
Framework)

Includes the base class for ant colony optimization, with a basic binary
representation.

Copyright Diego Diaz Fidalgo 2014/08/17

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import random
import math
import operator


class aco():
    """
    Basic Ant Colony Optimization
    """
    
    def __init__(self, sol_length, num_ants=50, default_ph=1,
                 evaporation=0.95, heuristics=None, num_ants_ph=3, elitism=1,
                 alpha=1, beta=1, max_iter=1000, rand_seed=None, rand_offset=0):
        """
        Initializaton of the ACO:
        The following parametres are needed (defaults will be used if no value
        is provided, except for the solution length)
        - Solution length, number of bits
        - Number of ants per iteration
        - Default pheromone level for matrix initialization
        - Evaporation: the factor to appl for pheromone evaporation (e.g. the
          0.8 default means 20% evaporation)
        - Heuristics: the values to use as heuristics when determining
          candidates. Make sure to use the format that you expect in your
          ant() method. For the default, heuristics[x][b] is the value
          associated to value b in position x (list of lists)
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
        self._sol_length = sol_length
        self._num_ants = num_ants
        self._pop = []
        self._default_ph = default_ph
        self._evaporation = evaporation
        self.init_ph()
        self._num_ants_ph = num_ants_ph
        self._heuristics = heuristics
        self._num_iters = 0
        self._best_obj = None
        self._best_sol = None
        self._elitism = elitism
        self._alpha = alpha
        self._beta = beta
        self._max_iter = max_iter
        self._random = random.Random()
        self._random.seed(rand_seed)
        if rand_offset:
            self._random.jumpahead(rand_offset)
        # Define the operations needed for calculating the pheromone and
        # heuristic impacts for candidate selection
        self._pow_alpha = lambda x: pow(x, self._alpha)
        self._pow_beta = lambda x: pow(x, self._beta)
        # Incoming population: if exists, add to the self-generated population
        self._incoming_population = []

    def init_ph(self):
        """
        Initialize pheromone matrix

        Since the representation of the pheromone is problem specific, override
        this method to provide a suitable initialization. This is called from
        self.__init__() during instatiation.
        """
        # for each element in the solution, there is a list with the pheromone
        # associated to 0 and 1 as values: pheromones[x][b] is the pheromone
        # corresponding to b in position x of the solution
        self._pheromones = [[self._default_ph] * 2 for _ in
                            range(self._sol_length)]
        self._base = [pow(2, n) for n in range(self._sol_length)]

    def ant(self):
        """
        Behaviour of each ant.

        This method builds a (randomized) solution based on the pheromone and
        heuristics matrices and returns it.

        This version expects each matrix to be a list of lists where
        matrix[x][b] is the value associated to value b at position x of the
        solution.

        The solution building process randomly decides a value for each position
        with probabilities proportional to s_i = (p_i)^a * (h_i)^b. p_i is the
        pheromone for value i in the position, and h_i is the heuristic for
        value i in the position. The s_i are normalized into probabilities by
        dividing by the sum of all s_i.

        If no euristics are available, they are ignored (equivalent to all
        heuristic values being 1).
        
        Override this method to customize your solution building procedure.
        """
        mysol = []
        for s in range(self._sol_length):
            pherom_probs = (self._pow_alpha(pheromone)
                            for pheromone in self._pheromones[s])
            if self._heuristics:
                heuristic_probs = (self._pow_beta(heur)
                                   for heur in self._heuristics[s])
                probs = [pherom_prob * heur_prob
                         for pherom_prob, heur_prob in zip(pherom_probs,
                                                           heuristic_probs)]
            else:
                probs = list(pherom_probs)
            sum_probs = sum(probs)
            rnd_num = self._random.uniform(0, sum_probs)
            select = -1
            cum_prob = 0
            while cum_prob <= rnd_num:  # Always goes in at least once
                select += 1
                cum_prob += probs[select]
            mysol.append(select)
        return mysol

    def _rank_pop(self, pop):
        """
        Rank a population based on the fitness value

        Return a list with the elements of pop sorted by fitness in ascending
        order. The list contains (individual, fitness) tuples.

        Override this method to provide a diffrent ranking mechanism
        (e.g. for maximization or for multiobjective)
        """
        # Decorate - sort - undecorate pattern
        decorated_pop = list(zip(pop, map(self._fitness, pop)))
        decorated_pop.sort(key=operator.itemgetter(1))
        return decorated_pop


    def _fitness(self, indiv):
        """
        Fitness function

        Return the fitness value associated to a given individual. Lower values
        are better (fitter), since the assumption is a minimization problem.

        This particular example returns the sum of the values.

        Override this method to generate a suitable fitness function for a
        different representation
        """
        return sum(indiv)

    def _end_condition(self):
        """
        Determine whether the algorithm is considered finished.

        The end condition here is number of iterations >= max_iter

        Override this method to generate a suitable end condition for a
        different problem or representation
        """
        return self._num_iters >= self._max_iter

    def _update_ph(self, pop):
        """
        Update the pheromone matrix for each (sol, fitness) tuple in population

        The update is by 1 /(1 + fitness) to avoid problems with division by
        zero.

        Override this method to use a different pheromone update scheme.
        """
        for sol, fit in pop[:self._num_ants_ph]:
            ph = 1.0 / (1 + fit)
            for s in range(self._sol_length):
                self._pheromones[s][sol[s]] += ph

    def _evaporate(self):
        """
        Apply evaporation

        Override this to adapt it to your own pheromone matrix implementation
        """
        for s in range(self._sol_length):
            for p in [0, 1]:  # Implementation specific for the binary values
                self._pheromones[s][p] *= self._evaporation
    
    def _run_iteration(self):
        """
        Execute and iteration of the ACO
        """
        pop = [self.ant() for _ in range(self._num_ants)]
        if self._incoming_population:
            pop.extend(self._incoming_population)
        ranked_pop = self._rank_pop(pop)
        pheromone_ants = ranked_pop[:self._num_ants_ph]
        if self._elitism:
            pheromone_ants.extend(self._pop)
            self._pop.extend(ranked_pop[:self._elitism])
            self._pop.sort(key=operator.itemgetter(1))
            self._pop = self._pop[:self._elitism]
        self._update_ph(pheromone_ants)
        new_best_sol = pheromone_ants[0]
        if self._best_obj is None or self._best_obj > new_best_sol[1]:
            self._best_sol, self._best_obj = new_best_sol
        self._evaporate()
        self._num_iters += 1

    def _run(self):
        while not self._end_condition():
            self._run_iteration()
            if self._num_iters % 10 == 0:
                print("Iteration: {it}, best obj: {obj}".format(
                      it=self._num_iters, obj=self._best_obj))


if __name__ == "__main__":
    sol_len = 50
    heur = [[1.0, 0.5] for _ in range(sol_len)]
    #heur = None
    max_it = 100
    myaco = aco(sol_length=sol_len, max_iter=max_it, heuristics=heur)
    myaco._run()
    print(myaco._best_sol, myaco._best_obj)
    print(myaco._pheromones)
