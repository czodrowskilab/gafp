from enum import Enum
import numpy as np
import random
from numpy.random import randint
import settings


class Mutate_functions:
    settings = settings.mutation_settings.get(settings.default_mutation_setting)

    @classmethod
    def setMutationRate(cls,**kwargs):
        cls.settings.update(kwargs)

    @classmethod
    def simple1(cls,entity,**kwargs):
        args = cls.settings.copy()
        args.update(kwargs)
        entity.mutate(rate = args.get("mutation_rate"),
                      strength = args.get("mutation_strength"),
                      layer_drop_rate = args.get("layer_drop_rate"),
                      layer_add_rate = args.get("layer_drop_rate"))

    functions = {"simple1": simple1}
    default = simple1
    pass


class Crossover_functions:
    rate = 0.0
    size = 1
    settings = settings.mutation_settings.get(settings.default_mutation_setting)

    @classmethod
    def setCrossoverRate(cls, **kwargs):
        cls.settings.update(kwargs)

    @classmethod
    def simple1(cls,elder1,elder2,**kwargs):
        kwargs.update(cls.settings)
        chromosomes_len = min(len(elder1.genome.chromosomes)-1,len(elder2.genome.chromosomes)-1)
        if chromosomes_len > 1 and random.random()< kwargs.get("crossing_over_rate"):
            cross_point = randint(1,chromosomes_len)

            cc1 = [elder1.genome.chromosomes[i].copy() for i in range(0, cross_point)]
            cc2 = [elder2.genome.chromosomes[i].copy() for i in range(0, cross_point)]

            cc1 += [elder2.genome.chromosomes[i].copy() for i in range(cross_point, len(elder2.genome.chromosomes))]
            cc2 += [elder1.genome.chromosomes[i].copy() for i in range(cross_point, len(elder1.genome.chromosomes))]

            c1 = elder1.copy()
            c2 = elder2.copy()

            avg_duration = round((elder1.duration + elder2.duration)/2)

            two_dropouts = False
            for i in range(1,len(cc1)):
                if cc1[i-1].layer_type=='dropout' and cc1[i].layer_type=='dropout':
                    two_dropouts = True

            for i in range(1,len(cc2)):
                if cc2[i-1].layer_type=='dropout' and cc2[i].layer_type=='dropout':
                    two_dropouts = True

            if not two_dropouts:
                c1.genome.chromosomes = cc1
                c1.genome.calc_penalty()
                c1.score = 0.00
                c1.status = (0,0)
                c1.duration = avg_duration

                c2.genome.chromosomes = cc2
                c2.genome.calc_penalty()
                c2.score = 0.00
                c2.status = (0, 0)
                c2.duration = avg_duration

            c1.best = False
            c2.best = False
            return [c1,c2]
        else:
            c1 = elder1.copy()
            c1.best = False
            c2 = elder2.copy()
            c2.best = False
            return [c1,c2]

    functions = {"simple1":
                 simple1}
    default = simple1


class EvolutionStrategies:
    strategy_functions = {}

    class strategies(Enum):
        plus1 = 1
        plus2 = 2
        parent_plus_child = 3
        only_children = 4
        drop_worst_percent = 5
        only_mutation = 6

    strategy = strategies.only_children
    mutate_func = Mutate_functions.default
    crossover_func = Crossover_functions.default
    evaluation_func = None
    settings = settings.mutation_settings.get(settings.default_mutation_setting)

    @classmethod
    def register(cls,enum_entry,func):
        cls.strategy_functions[enum_entry] = func

    @classmethod
    def plus1(cls,population):
        cls.mutate_func(population)
        cls.crossover_func(population)
        pass

    @classmethod
    def parent_plus_child(cls,population,**kwargs):
        params = {'replicate_rate':0.5}
        params.update(kwargs)

        target_size = len(population)
        repl_size = int(target_size * params['replicate_rate'])

        # Step 1: Select parents for replication and duplicate them
        parents_for_replication = np.random.choice(population,size=repl_size,replace=False)
        children = population.copy()
        child_entities = list(map(lambda x: x.copy(),parents_for_replication))
        children.population = child_entities

        # Step 2: Mutate children
        ch = map(lambda x: cls.mutate_func(x), children)

        # Step 3: Evaluate children, Sort all by score and select target_size for new generation
        cls.evaluation_func(children)
        intermediate_pop = population + children
        intermediate_pop.sort()
        new_pop = intermediate_pop[:target_size]

        evolve_meta = {}
        return new_pop, evolve_meta

    @classmethod
    def only_children(cls, population, **kwargs):
        params = {'replicate_rate': 1.0}
        params.update(kwargs)

        target_size = len(population)
        repl_size = int(target_size * params['replicate_rate'])

        if target_size % 2 != 0:
            raise Exception("target-size of population should be even, it is {}".format(target_size))


        parents_for_replication = np.random.choice(population, size=repl_size, replace=False)
        parents_for_replication.shape = (int(target_size/2),2)

        child_entities = []
        for x in parents_for_replication:
            new_entities = cls.crossover_func(x[0],x[1])
            for y in new_entities:
                child_entities.append(y)

        children = child_entities
        m = list(map(lambda x: cls.mutate_func(x), children))

        evolve_meta = {}
        return children, evolve_meta

    @classmethod
    def only_mutation(cls,population,**kwargs):
        cls.settings.update(kwargs)
        Mutate_functions.rate = 0.1

        m = list(map(lambda x: cls.mutate_func(x), population[params['save_best_n']:]))
        evolve_meta = {}
        return population, evolve_meta

    @classmethod
    def drop_worst_percent(cls, population, **kwargs):
        kwargs.update(cls.settings)

        Crossover_functions.rate = kwargs.get('crossing_over_rate')
        Mutate_functions.rate = kwargs.get('mutation_rate')
        Mutate_functions.strength = [kwargs.get('mutation_strength'), kwargs.get('mutation_strength')]

        target_size = len(population)
        drop_size = max([int(target_size * kwargs.get('evolution_strategy_percentage')), 2])
        if drop_size % 2 != 0:
            drop_size -= 1

        while((target_size - drop_size) < 2):
            print("drop-size {} too high for a population of {}... decrease drop_size".format(drop_size,target_size))
            drop_size -= 1
            if drop_size < 1:
                break
        print("drop {} worst entities of population".format(drop_size))
        if drop_size > 0:
            new_pop = population[:-drop_size]
        else:
            new_pop = population[:]

        parents_for_replication = np.random.choice(new_pop, size=drop_size, replace=True) # we want some entities beeing parants of multiple children
        parents_for_replication.shape = (int(drop_size / 2), 2)

        for x in parents_for_replication:
            new_entities = cls.crossover_func(x[0], x[1])
            for y in new_entities:
                if len(new_pop) >= target_size:
                    break
                new_pop.append(y)

        # we want to leave the best untouched
        m = list(map(lambda x: cls.mutate_func(x), new_pop[1:]))
        evolve_meta = {}
        return new_pop, evolve_meta

    @classmethod
    def set_strategy(cls,strategy):
        if not strategy in cls.strategies:
            raise Exception("strategy {} not in strategies-enum: {}".format(strategy,cls.strategies))
        cls.strategy = strategy
        pass

    @classmethod
    def set_settings(cls, **kwargs):
        cls.settings.update(kwargs)

    @classmethod
    def set_mutate_func(cls,mutate_func):
        cls.mutate_func = mutate_func

    @classmethod
    def set_crossover_func(cls,crossover_func):
        cls.crossover_func = crossover_func

    @classmethod
    def evolve(cls,population):
        new_pop,evolve_meta = cls.strategy_functions[cls.strategy](population)
        return new_pop,evolve_meta

EvolutionStrategies.register(EvolutionStrategies.strategies.plus1, EvolutionStrategies.plus1)
EvolutionStrategies.register(EvolutionStrategies.strategies.parent_plus_child, EvolutionStrategies.parent_plus_child)
EvolutionStrategies.register(EvolutionStrategies.strategies.only_children, EvolutionStrategies.only_children)
EvolutionStrategies.register(EvolutionStrategies.strategies.drop_worst_percent, EvolutionStrategies.drop_worst_percent)
EvolutionStrategies.register(EvolutionStrategies.strategies.only_mutation, EvolutionStrategies.only_mutation)

