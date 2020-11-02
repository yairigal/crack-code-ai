import json
import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import logging
import random
import matplotlib.pyplot as plt
import numpy as np

from ai_player import AIPlayer
from game import CrackCodeGame

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(sh)


class CrackCodeGeneticAlgorithm:

    def __init__(self):
        self.population_size = 100
        self.selection_percentage = 0.4
        self.max_parents_amount = 4
        self.mutation_rate = 0.1
        self.mutation_range = 0.1

        self.current_generation = 0
        self.best_model: AIPlayer = None

        self.population: List[AIPlayer] = None
        self.avg_fitnesses = []
        self.top_fitnesses = []

    @staticmethod
    def _individual_fitness(results, game):
        total_hints_normalized = results['total_hints'] / game.max_hints
        turns_normalized = results['turns'] / game.turns
        won = results['won']

        if won:
            return game.turns / results['turns']

        else:
            hits = sum(secret == guess for secret, guess in zip(game.secret_code, game.game_output[-1].guess))
            hits_normalized = hits / game.code_length  # 0.25, 0.5, 0.75, 1

            return hits_normalized + 5 * total_hints_normalized

    def fitness_test(self):
        fitnesses = []
        for i, individual in enumerate(self.population):
            print(f'fitness test {i + 1}/{self.population_size}', end='\r')
            game = CrackCodeGame(individual)
            results = game.run()
            fitnesses.append(self._individual_fitness(results, game))
            del game

        return fitnesses

    def selection(self, fitnesses):
        self.population, fitnesses = zip(*sorted(zip(self.population, fitnesses), key=lambda i: i[1], reverse=True))
        total_fitness = sum(fitnesses)
        selection_amount = int(self.selection_percentage * self.population_size)

        self.best_model = self.population[0]
        self.average_fitness = total_fitness / self.population_size
        self.top_fitnesses.append(fitnesses[0])
        self.avg_fitnesses.append(self.average_fitness)
        if total_fitness == 0:
            # All are the same
            return [random.choice(self.population) for _ in range(selection_amount)]

        fitnesses_normalized = [fitness / total_fitness for fitness in fitnesses]
        fitness_accumulator = 0
        fitnesses_accumulated = []
        for fitness in fitnesses_normalized:
            fitness_accumulator += fitness
            fitnesses_accumulated.append(fitness_accumulator)

        selected = []
        for _ in range(selection_amount):
            for individual, fitness_accumulated in zip(self.population, fitnesses_accumulated):
                if random.random() <= fitness_accumulated:
                    selected.append(individual)
                    break

        assert len(selected) == selection_amount
        return selected

    @staticmethod
    def _individual_crossover(*parents):
        weights = [parent.model.get_weights() for parent in parents]

        child_weights = np.mean(np.array(weights), axis=0)
        child = AIPlayer()

        child.model.set_weights(list(child_weights))
        return child

    def crossover(self, selected):
        population_size_left = self.population_size - len(selected)
        added_population = []
        for _ in range(population_size_left):
            parents_amount = random.randint(2, self.max_parents_amount)
            parents = [random.choice(selected) for _ in range(parents_amount)]
            added_population.append(self._individual_crossover(*parents))

        selected.extend(added_population)
        return selected

    def _individual_mutation(self, individual: AIPlayer):
        weights = individual.model.get_weights()
        weights = np.array(weights)
        weights += np.random.uniform(0, self.mutation_range, weights.shape)
        individual.model.set_weights(list(weights))

    def mutation(self):
        for individual in self.population:
            if random.random() <= self.mutation_rate:
                self._individual_mutation(individual)

    def generation(self):
        # fitness test
        logger.debug('fitness test')
        fitnesses = self.fitness_test()
        # selection
        logger.debug("selection")
        selected_individuals = self.selection(fitnesses)
        # crossover
        logger.debug("crossover")
        self.population = self.crossover(selected_individuals)
        # mutation
        logger.debug("mutation")
        self.mutation()

    def display_summary(self):
        print(f"Generation {self.current_generation}:\n"
              f"average fitness={self.average_fitness}")

        plt.plot(list(range(self.current_generation)), self.avg_fitnesses, color='b', label='average fitnesses')
        plt.plot(list(range(self.current_generation)), self.top_fitnesses, color='r', label='top fitnesses')
        plt.pause(0.000001)

    def _init_population(self):
        return [AIPlayer() for _ in range(self.population_size)]

    def run(self, skip_init=False):
        if not skip_init:
            self.population = self._init_population()

        plt.plot(list(range(self.current_generation)), self.avg_fitnesses, color='b', label='average fitnesses')
        plt.plot(list(range(self.current_generation)), self.top_fitnesses, color='r', label='top fitnesses')
        plt.legend(loc="upper left")
        plt.xlabel('generation')

        while True:
            self.current_generation += 1
            self.generation()
            self.display_summary()
            self._save_best_model()
            if self.current_generation % 10 == 0:
                self._checkpoint()

    def _save_best_model(self):
        logger.debug('Saving best model')
        self.best_model.model.save('./best_model')

    def _checkpoint(self):
        if not os.path.exists('./checkpoint'):
            os.mkdir('./checkpoint')
        # save population
        if not os.path.exists('./checkpoint/population'):
            os.mkdir('./checkpoint/population')

        for i, indv in enumerate(self.population):
            indv.pickle(f'./checkpoint/population/individual_{i}')

        # save other stuff
        with open('./checkpoint/instance.data', 'w') as f:
            json.dump({
                'population_size': self.population_size,
                'selection_percentage': self.selection_percentage,
                'max_parents_amount': self.max_parents_amount,
                'mutation_rate': self.mutation_rate,
                'mutation_range': self.mutation_range,
                'current_generation': self.current_generation,
                'avg_fitnesses': self.avg_fitnesses,
                'top_fitnesses': self.top_fitnesses
            }, f)

    @classmethod
    def load_checkpoint(cls):
        instance = cls()
        with open('./checkpoint/instance.data', 'r') as f:
            instance.__dict__.update(json.load(f))

        instance.population = []
        for indv in os.listdir('./checkpoint/population'):
            player = AIPlayer.unpickle(f'./checkpoint/population/{indv}')
            instance.population.append(player)

        return instance


if __name__ == '__main__':
    if os.path.exists('./checkpoint'):
        logger.info('loading checkpoint')
        game = CrackCodeGeneticAlgorithm.load_checkpoint()
        game.run(skip_init=True)

    else:
        CrackCodeGeneticAlgorithm().run()
