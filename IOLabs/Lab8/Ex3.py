import math
import pygad


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


gene_type = {"low": 0.0, "high": 1.0}


def fitness_func(model, solution, solution_idx):
    metal_endurance = endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])
    return metal_endurance


fitness_function = fitness_func

chromosome_population = 10
genes_number = 6
parents_propagation_number = 3
number_of_generations = 20
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 17

ga_instance = pygad.GA(gene_space=gene_type,
                       num_generations=number_of_generations,
                       num_parents_mating=parents_propagation_number,
                       fitness_func=fitness_function,
                       sol_per_pop=chromosome_population,
                       num_genes=genes_number,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
ga_instance.plot_fitness()
