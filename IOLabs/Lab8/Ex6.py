import matplotlib.pyplot as plt
import random

from aco import AntColony

plt.style.use("dark_background")

coordinates = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (12, 48),
    (33, 34),
    (56, 39)
)


def random_coord():
    r = random.randint(0, len(coordinates))
    return r


def plot_nodes(w=12, h=8):
    for x, y in coordinates:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in coordinates for b in coordinates)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(coordinates, ant_count=300, alpha=0.25, beta=1.5,
                    pheromone_evaporation_rate=0.30, pheromone_constant=1000.0,
                    iterations=5)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.show()
