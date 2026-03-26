import pygad
import numpy

maze_matrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1], [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# Player movement in maze: 0 - up, 1 - right, 2 - down, 3 - left
gene_type = [0, 1, 2, 3]

chromosome_length = 30
chromosome_population = 20
parents_propagation_number = 15
keep_parents = 4
mutation_percent_genes = 7
number_of_generations = 1000
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"


def player_move(x, y, move):
    player_pos = (x, y)
    if move == 0:
        player_pos = (x - 1, y)
    elif move == 1:
        player_pos += (x, y + 1)
    elif move == 2:
        player_pos += (x + 1, y)
    elif move == 3:
        player_pos += (x, y - 1)
    return player_pos


def fitness_func(model, solution, solution_idx):
    start_pos = maze_matrix[1][1]
    end_pos = maze_matrix[10][10]
    player_pos = (1, 1)
    for command in solution:
        new_pos = player_move(player_pos[0], player_pos[1], command)
        if maze_matrix[new_pos[0]][new_pos[1]] == 0:
            player_pos = new_pos
            continue
        else:
            fitness = 0
            return fitness

    fitness = numpy.abs(player_pos[0] - 10) + numpy.abs(player_pos[1] - 10)
    return fitness


fitness_function = fitness_func

ga_instance = pygad.GA(gene_space=gene_type,
                       num_generations=number_of_generations,
                       num_parents_mating=parents_propagation_number,
                       fitness_func=fitness_function,
                       sol_per_pop=chromosome_population,
                       num_genes=chromosome_length,
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
