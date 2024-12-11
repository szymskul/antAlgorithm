import numpy as np
import matplotlib.pyplot as plt
import random
import Ant

def read_data(file_path):
    nodes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            nr = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            nodes.append((nr, x, y))
    return nodes


def calculate_distance_matrix(nodes):
    size = len(nodes)
    distance_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                xi, yi = nodes[i][1], nodes[i][2]
                xj, yj = nodes[j][1], nodes[j][2]
                distance = np.hypot(xi - xj, yi - yj)
                if distance == 0.0:
                    distance = 0.000001
                distance_matrix[i][j] = distance
    print(distance_matrix)
    return distance_matrix


def aco_tsp(nodes, num_ants, num_iterations, alpha, beta, evaporation_rate, random_choose_rate):
    size = len(nodes)
    distance_matrix = calculate_distance_matrix(nodes)
    pheromones = np.ones((size, size))
    heuristic = 1 / (distance_matrix + np.diag([np.inf] * size))  # Avoid division by zero

    best_length = np.inf
    best_tour = None

    for iteration in range(num_iterations):
        ants = [Ant.Ant(random.randint(0, size - 1)) for _ in range(num_ants)]
        for ant in ants:
            ant.visited.add(ant.start_node)
            while len(ant.visited) < size:
                current = ant.current_node
                unvisited = list(set(range(size)) - ant.visited)
                if random.random() < random_choose_rate:
                    next_node = random.choice(unvisited)
                else:
                    probabilities = []
                    for j in unvisited:
                        tau = pheromones[current][j] ** alpha
                        eta = (1 / heuristic[current][j]) ** beta
                        probabilities.append(tau * eta)
                    total = sum(probabilities)
                    probabilities = [p / total for p in probabilities]
                    next_node = random.choices(unvisited, weights=probabilities, k=1)[0]
                ant.visit_node(next_node, distance_matrix[current][next_node])
            if ant.length < best_length:
                best_length = ant.length
                best_tour = ant.tour

        # Evaporate pheromones
        pheromones -= pheromones * evaporation_rate

        # Deposit pheromones
        for ant in ants:
            for edge in ant.tour:
                pheromones[edge[0]][edge[1]] += 1 / ant.length
                pheromones[edge[1]][edge[0]] += 1 / ant.length  # Assuming undirected graph

        print(f"Iteration {iteration + 1}/{num_iterations}, Best Length: {best_length}")

    return best_tour, best_length


def plot_tour(nodes, tour):
    x = [nodes[edge[0]][1] for edge in tour]
    y = [nodes[edge[0]][2] for edge in tour]

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, 'o-', color='blue')

    for node in tour:
        plt.text(nodes[node[0]][1], nodes[node[0]][2], str(nodes[node[0]][0]))

    plt.title("Najkrótsza Znaleziona Trasa")
    plt.xlabel("Współrzędna X")
    plt.ylabel("Współrzędna Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = 'A-n32-k5.txt'
    nodes = read_data(file_path)
    best_tour, best_length = aco_tsp(
        nodes,
        num_ants=10,
        num_iterations=1000,
        alpha=1.0,
        beta=1.0,
        evaporation_rate=0.1,
        random_choose_rate=0.3
    )
    print("Najlepsza długość trasy:", best_length)
    plot_tour(nodes, best_tour)
