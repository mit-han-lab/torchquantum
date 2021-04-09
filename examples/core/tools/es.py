import numpy as np
import random


__all__ = ['SimpleGA', 'EvolutionEngine']


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid,
                                    axis=1)


class SimpleGA(object):
    """Simple Genetic Algorithm."""
    def __init__(self, num_params,      # number of model parameters
                 sigma_init=0.1,        # initial standard deviation
                 sigma_decay=0.999,     # anneal standard deviation
                 sigma_limit=0.01,      # stop annealing if less than this
                 popsize=256,           # population size
                 elite_ratio=0.1,       # percentage of the elites
                 forget_best=False,     # forget the historical best elites
                 weight_decay=0.01,     # weight decay coefficient
                 ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.epsilon = None
        self.solutions = None
        self.curr_best_reward = None

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        """returns a list of parameters"""
        self.epsilon = np.random.randn(self.popsize,
                                       self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand(c.size) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a],
                                self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert(len(reward_table_result) == self.popsize), \
            "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def result(self):
        # return best params so far, along with historically best reward,
        # curr reward, sigma
        return (self.best_param, self.best_reward,
                self.curr_best_reward, self.sigma)


class Converter(object):
    def __init__(self, n_wires, n_available_wires, arch_space):
        self.n_wires = n_wires
        self.n_available_wires = n_available_wires
        self.arch_space = arch_space

    @staticmethod
    def solution2gene(solution):
        return solution['layout'] + solution['arch']

    def gene2solution(self, gene):
        return {'layout': gene[:self.n_wires], 'arch': gene[self.n_wires:]}

    def get_gene_choice(self):
        return [list(range(self.n_available_wires))] * self.n_wires + \
            self.arch_space


class EvolutionEngine(object):
    def __init__(self,
                 population_size,
                 parent_size,
                 mutation_size,
                 mutation_prob,
                 crossover_size,
                 n_wires,
                 n_available_wires,
                 arch_space,
                 gene_mask=None,
                 ):
        self.population_size = population_size
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.mutation_prob = mutation_prob
        self.crossover_size = crossover_size
        assert self.population_size == self.parent_size + self.mutation_size \
               + self.crossover_size

        self.n_wires = n_wires
        self.n_available_wires = n_available_wires
        self.arch_space = arch_space

        self.converter = Converter(
            self.n_wires,
            self.n_available_wires,
            self.arch_space
        )

        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)
        self.best_solution = None
        self.best_score = None

        # to constraint the design space in a fine-grained manner
        self.gene_mask = gene_mask

        # initialize with random samples
        self.population = self.random_sample(self.population_size)

    def ask(self):
        """return the solutions"""
        return [self.converter.gene2solution(gene) for gene in self.population]

    def tell(self, scores):
        """perform evo search according to the scores"""
        sorted_idx = np.array(scores).argsort()[:self.parent_size]
        self.best_solution = self.converter.gene2solution(self.population[
                                                           sorted_idx[0]])
        parents = [self.population[i] for i in sorted_idx]
        self.best_score = scores[sorted_idx[0]]

        # mutation
        mutate_population = []
        k = 0
        while k < self.mutation_size:
            mutated_gene = self.mutate(random.choices(parents)[0])
            mutated_gene = self.apply_gene_mask(mutated_gene)
            if self.satisfy_constraints(mutated_gene):
                mutate_population.append(mutated_gene)
                k += 1

        # crossover
        crossover_population = []
        k = 0
        while k < self.crossover_size:
            crossovered_gene = self.crossover(random.sample(parents, 2))
            crossovered_gene = self.apply_gene_mask(crossovered_gene)
            if self.satisfy_constraints(crossovered_gene):
                crossover_population.append(crossovered_gene)
                k += 1

        self.population = parents + mutate_population + crossover_population

    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])
        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(self.gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])
        return mutated_gene

    def satisfy_constraints(self, gene):
        # Different logical qubits map to different physical qubits
        return len(set(gene[:self.n_wires])) == self.n_wires

    def apply_gene_mask(self, sample_gene):
        masked_gene = sample_gene.copy()
        if self.gene_mask is not None:
            for k, gene in enumerate(self.gene_mask):
                if not gene == -1:
                    masked_gene[k] = gene

        return masked_gene

    def random_sample(self, sample_num):
        population = []
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                samp_gene.append(random.choices(self.gene_choice[k])[0])
            samp_gene = self.apply_gene_mask(samp_gene)
            if self.satisfy_constraints(samp_gene):
                population.append(samp_gene)
                i += 1

        return population
