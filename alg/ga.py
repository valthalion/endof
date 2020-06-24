"""
Genetic Algorithm class for ENDOF (Endof New Distributed Optimiaztion Framework)

Includes the base class for genetic algorithms, with a basic binary gene
representation and simple random crossover and mutation operations.

Copyright Diego Diaz Fidalgo 2014/07/27

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import random
import math
import operator


class ga():
    """
    Basic Genetic Algorithm 
    """
    def __init__(self, num_genes, pop_size=50, elitism=0, crossover_prob=0.5,
                 mutation_prob=0.05, max_iter=1000, rand_seed=None,
                 rand_offset=0):
        """
        Initialization of the GA:
        The following parametres are needed (defaults will be used if no
        value is provided, except for the number of genes)
        - Number of Genes in an individual
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
        # TODO: Find good default values for params
        self._num_genes = num_genes
        self._pop_size = pop_size
        self._pop = []
        self._num_iters = 0
        self._best_obj = None
        self._best_sol = None
        self._elitism = elitism
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._max_iter = max_iter
        self._random = random.Random()
        self._random.seed(rand_seed)
        if rand_offset:
            self._random.jumpahead(rand_offset)
        # Incoming population: if exists, add to the self-generated population
        self._incoming_population = []


    def initialize_population(self):
        """
        Initialize the poulation

        Create a random initial population for the GA. This generates an integer
        in the right range for each individual. The integers will actually be
        interpreted as bit arrays.

        Override this method to generate a suitable population for a different
        representation
        """
        # Convert number of genes to the largest integer under the binary
        # representation with that number of bits.
        # Do this here instead of in self.__init__() because this is specific to
        # the representation, and this method should be overridden when
        # subclassing
        self._indiv_size = pow(2, self._num_genes) - 1
        pop = [self._random.randint(0, self._indiv_size) for _ in
               range(self._pop_size)]
        self._pop = self._rank_pop(pop)
    
    def _rank_pop(self, pop):
        """
        Rank a population based on the fitness value
        
        Return a list of tuples (individual, fitness) sorted by fitness in
        ascending order.
        
        The individuals for the population are those provided in pop.
        
        Override this method to provide a diffrent ranking mechanism
        (e.g. for maximization or for multiobjective)
        """
        decorated_pop = zip(pop, map(self._fitness, pop))
        decorated_pop.sort(key=operator.itemgetter(1))
        if len(decorated_pop) > self._pop_size:
            return decorated_pop[:self._pop_size]
        else:
            return decorated_pop
    
    def _crossover(self, parent1, parent2):
        """
        Crossover operation
        
        Given two parent individuals, generate a new one containing the upper
        bits of parent1 and the lower bits of parent2. The splitting point is
        randomly generated.
        
        Override this method to generate a suitable crossover operation for a
        different representation
        """
        # The limits for randint are defined so that at least one element from
        # each parent remains: at the lowest the split will be at position 1,
        # and at the highest at n-1 for n bits (indiv_size = 2^n - 1)
        split_bit = \
            math.floor(math.log(self._random.randint(2, self._indiv_size), 2))
        split_lower_mask = int(pow(2, split_bit) - 1)
        split_upper_mask = self._indiv_size - split_lower_mask
        return (parent1 & split_upper_mask) + (parent2 & split_lower_mask)
    
    def _apply_crossover(self, pop):
        """
        Application of the crossover operation to the selected parents.
        
        The probability of selection for an element in the provided population
        is proportional to the inverse of its fitness plus one (to avoid
        problems with division by zero).
        
        Override this method to chose a different way to apply crossover.
        """
        # Probabilities here should strictly be divided by the overall sum, but
        # instead a random number is generated in U(0, sum).
        # Probabilities associated to each individual
        inv_ranks = [1.0 / (indiv[1] + 1) for indiv in pop]
        # Total probability
        total_prob = sum(inv_ranks)
        # Each element is (p_i, i),
        # where i is an individual and p_i its probability
        decorated_pop = zip(inv_ranks, pop)
        
        children_pop = []
        for _ in range(self._pop_size):
            # Random number for the selection of the first parent pval1
            # The selected individual is the first for which the accumulated
            # probability is >= the random number
            pval1 = self._random.uniform(0, total_prob)
            acc_prob = 0
            idx1 = 0
            while decorated_pop[idx1][0] + acc_prob < pval1:
                acc_prob += decorated_pop[idx1][0]
                idx1 += 1
            # The probability accumulator contains the accumulated probability
            # up to and including idx1 - 1
            # Remove the part corresponding to the individual at idx1 from the
            # total and generate the new random number in the new uniform range
            pval2 = self._random.uniform(0, total_prob - decorated_pop[idx1][0])

            # pval2 > accumulator means that we can skip the individuals before
            # idx1, and idx1 itself, as we are considering it removed (and is
            # not included in the total probability). We start from idx1 + 1,
            # with the existing value in the accumulator, which is equivalent to
            # the approach used for idx1 with the individual at idx1 removed,
            # but idx2 will still refer to the index of the selected individual
            # without actually removing the one at idx1
            if pval2 > acc_prob:
                idx2 = idx1 + 1
            # The new selected individual will appear before idx1, so we can
            # repeat exactly the same process as for idx1
            else:
                acc_prob = 0
                idx2 = 0
            # The loop is the same for both conditions, with the updated index
            # and accumulator
            while decorated_pop[idx2][0] + acc_prob < pval2:
                acc_prob += 1
                idx2 += 1

            # Add the result of the crossover to the list of children
            # Pass just the individual, not the fitness (the final [0])
            parent1 = decorated_pop[idx1][1][0]
            parent2 = decorated_pop[idx2][1][0]
            children_pop.append(self._crossover(parent1, parent2))
            children_pop.append(self._crossover(parent2, parent1))
        return children_pop
    
    
    def _mutation(self, indiv):
        """
        Mutation operation
        
        Given an individual, generate a new one resulting from toggling one bit
        at a random location.
        
        Override this method to generate a suitable mutation operation for a
        different representation
        """
        # Select one position at random
        mutation_bit = self._random.randint(0, self._num_genes - 1)
        # And toggle the bit xor-ing a mask of all 0s,
        # except for a 1 at the bit position
        mask = 1 << mutation_bit
        return indiv ^ mask
    
    def _apply_mutation(self, pop):
        """
        Application of the mutation operation to the selected parents.
        
        Each individual in the passed population is mutated --generating a new
        individual-- with probability mutation_prob.
        
        Override this method to chose a different way to apply mutation.
        """
        offspring = []
        for indiv in pop:
            if self._random.random() <= self._mutation_prob:
                offspring.append(self._mutation(indiv[0]))
        return offspring
    
    def _fitness(self, indiv):
        """
        Fitness function
        
        Return the fitness value associated to a given individual. Lower values
        are better (fitter), since the assumption is a minimization problem.
        
        This particular example returns the integer value of the chromosome.
        
        Override this method to generate a suitable fitness function for a
        different representation
        """
        return indiv
    
    
    def _end_condition(self):
        """
        Determine whether the algorithm is considered finished.
        
        The end condition here is number of iterations >= max_iter
        
        Override this method to generate a suitable end condition for a
        different problem or representation
        """
        return self._num_iters >= self._max_iter
    
    
    def _select_parents(self):
        """
        Select the individuals in the population that will breed to generate the
        next one.
        
        They are selected with probability inverse to their fitness + 1 until a
        subpopulation of ceil(pop_size * crossover_prob) size has been reached.
        
        The best individual is always included.
        
        Override this method to define a different selection mechanism.
        """
        # decorate with probabilities as 1 / (fitness + 1)
        candidates = [(1.0 / (indiv[1] + 1), indiv) for indiv in self._pop]
        selected = []
        
        # The best individual is always included
        # Undecorate to add the (fitness, indiv) tuple
        selected.append(candidates.pop(0)[1])
        
        # Add the rest to make the selection size
        # ceil(pop_size * crossover_prob)
        selection_size = int(math.ceil(self._pop_size * self._crossover_prob))
        total_prob = sum(x[0] for x in candidates)
        prob_decrement = 0
        for _ in range(selection_size - 1):  # There's one already in
            # Start from 0 and check the accumulated probability for each
            # individual
            # The selected individual is the first for which the accumulated
            # probability is more that the random number generated
            # The individual is removed (transferred to the selected list) and
            # the process is repeated. No additional calculations needed since
            # the individual is removed and the accumulation is done on the fly
            # in each iteration rather than using the accumulated values in the
            # decoration
            total_prob -= prob_decrement
            pval = self._random.uniform(0, total_prob)
            acc_prob = 0
            idx = 0
            while candidates[idx][0] + acc_prob < pval:
                acc_prob += candidates[idx][0]
                idx += 1
            # Undecorate to add the (indiv, fitness) tuple
            prob_decrement = candidates[idx][0]
            selected.append(candidates.pop(idx)[1])
        return selected
    
    def _run_iteration(self):
        """
        Execute a generation of the GA
        """
        parents = self._select_parents()
        newpop = []
        if self._incoming_population:
            newpop.extend(self._incoming_population)
        newpop.extend(self._apply_crossover(parents))
        newpop.extend(self._apply_mutation(parents))
        if self._elitism:
            newpop.extend(indiv for (indiv, _) in self._pop[:self._elitism])
        self._pop = self._rank_pop(newpop)
        gen_best_sol, gen_best_obj = self._pop[0]
        if gen_best_obj < self._best_obj or self._best_obj is None:
            self._best_obj, self._best_sol = gen_best_obj, gen_best_sol
        self._num_iters += 1
    
    def print_pop(self):
        print("Population")
        for indiv, fit in self._pop:
            print(indiv, fit)
        print()

    def _run(self):
        """
        Standard execution, mainly for testing
        """
        if not self._pop:
            self.initialize_population()
        while not self._end_condition():
            self._run_iteration()
            if self._num_iters % 100 == 0:
                print("Iteration {it}: best obj {obj}".format(
                                    it=self._num_iters, obj=self._best_obj))


if __name__ == "__main__":
    myga = ga(num_genes=10, elitism=2)
    myga.initialize_population()
    myga._run()
    print(myga._best_sol, myga._best_obj)
