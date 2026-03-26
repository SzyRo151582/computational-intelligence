import pygad
import numpy
import time

start = time.time()
print("Program started")

# data
item_names = ["clock", "landscape", "portrait", "radio", "laptop", "night lamp", "silver cutlery", "porcelain", "bronze figure", "leather bag", "vacuum cleaner"]
item_costs = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
item_weights = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]

# genes
gene_type = [0, 1]

#fitness
def fitness_func(model, solution, solution_idx):
    cost_sum = numpy.sum(solution * item_costs)
    weight_sum = numpy.sum(solution * item_weights)
    if weight_sum > 25:
        fitness = 0
    else:
        fitness = cost_sum
    return fitness


fitness_function = fitness_func

# gan parameters preparation
chromosome_population = 15
genes_number = len(item_names)
parents_propagation_number = int(len(item_names) / 2)
number_of_generations = 10
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

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
                       mutation_percent_genes=mutation_percent_genes,
                       # f)
                       stop_criteria="reach_1630")

ga_instance.run()

#show the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()

solution_list = solution.tolist()
item_index = 0

print("The items that a thief should take to the bag are:")
for gene in solution_list:
    if gene == 1:
        print(item_names[item_index])

    item_index += 1

print(f"Fitness value of the best solution = {solution_fitness}.")
end = time.time()
print(f"Program time: {end - start}")

# showing changes in performance over generations
ga_instance.plot_fitness()

# d) Best solution:
# landscape, portrait, laptop, silver cutlery, porcelain, leather bag

# e) Ten runs: 60% accuracy
# f) 0.01443028450012207
