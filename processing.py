
import cv2
import numpy as np
import math
import os
from scipy.interpolate import PchipInterpolator

# ========================== FUNCIONES AUXILIARES ==========================

def entropy_func(gray_image):
    values, counts = np.unique(gray_image, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))

def calculate_psnr(original, processed):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)

def spline_cubic_transform(image, x_nodes, y_nodes):
    x_nodes = np.clip(np.array(x_nodes), 0, 1)
    y_nodes = np.clip(np.array(y_nodes), 0, 1)
    spline = PchipInterpolator(x_nodes, y_nodes)

    img = image.astype(np.float32) / 255.0
    transformed = np.empty_like(img)
    for c in range(3):
        transformed[:, :, c] = spline(img[:, :, c])
    transformed = np.clip(transformed * 255, 0, 255).astype(np.uint8)
    return transformed

def fitnessFunction(image, x_nodes, y_nodes):
    corrected_img = spline_cubic_transform(image, x_nodes, y_nodes)
    corrected_gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    ent = entropy_func(corrected_gray)
    image_original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    psnr = calculate_psnr(image_original_gray, corrected_gray)
    penalty = (25 - psnr) ** 2 if psnr < 25 else 0
    return -(ent - 100 * penalty)

# ========================== CLASE GA ==========================

class GeneticAlgorithm:
    def __init__(self, image, pop_size=30, generations=70, eta_c=1, eta_m=20, pc=0.8, pm=0.6):
        self.pop_size = pop_size
        self.generations = generations
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.pc = pc
        self.pm = pm
        self.image = image
        self.num_nodes = 3

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            x_inner = np.sort(np.clip(np.random.rand(self.num_nodes), 0.05, 0.95))
            while len(np.unique(x_inner)) < self.num_nodes:
                x_inner = np.sort(np.clip(np.random.rand(self.num_nodes), 0.05, 0.95))
            y_inner = np.cumsum(np.abs(np.random.rand(self.num_nodes)))
            y_inner /= y_inner[-1]
            individual = np.concatenate([x_inner, y_inner])
            population.append(individual)
        return np.array(population)

    def tournament_selection(self, population, fitness):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        return population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

    def sbx_crossover(self, p1, p2):
        if np.random.rand() > self.pc:
            return p1.copy(), p2.copy()
        beta = np.empty_like(p1)
        for j in range(len(p1)):
            u = np.random.rand()
            beta[j] = (2 * u) ** (1 / (self.eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return self.repair_individual(c1), self.repair_individual(c2)

    def polynomial_mutation(self, x):
        if np.random.rand() > self.pm:
            return x
        delta = np.empty_like(x)
        for j in range(len(x)):
            u = np.random.rand()
            delta[j] = (2 * u) ** (1 / (self.eta_m + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
        mutated = x + delta * 0.2
        return self.repair_individual(mutated)

    def repair_individual(self, ind):
        x_inner = np.sort(np.clip(ind[:self.num_nodes], 0.05, 0.95))
        while len(np.unique(x_inner)) < self.num_nodes:
            x_inner = np.sort(np.clip(np.random.rand(self.num_nodes), 0.05, 0.95))
        y_inner = np.clip(ind[self.num_nodes:], 0, 1)
        y_inner = np.cumsum(np.abs(y_inner))
        y_inner = y_inner / y_inner[-1]
        return np.concatenate([x_inner, y_inner])

    def fitness_with_constraint(self, ind):
        eps = 1e-6
        x_inner = ind[:self.num_nodes]
        x_nodes = np.concatenate([[0.0], x_inner, [1.0]])
        if not np.all(np.diff(x_nodes) > eps):
            return 1e9
        y_inner = ind[self.num_nodes:]
        y_nodes = np.concatenate([[0.0], np.cumsum(np.abs(y_inner)) / np.sum(np.abs(y_inner)), [1.0]])
        return fitnessFunction(self.image, x_nodes, y_nodes)

    def evolve(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.generations):
            fitness = np.array([self.fitness_with_constraint(ind) for ind in population])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx].copy()
                best_fitness = fitness[best_idx]
            new_population = []
            for _ in range(self.pop_size // 2):
                p1 = self.tournament_selection(population, fitness)
                p2 = self.tournament_selection(population, fitness)
                c1, c2 = self.sbx_crossover(p1, p2)
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)
                new_population.extend([c1, c2])
            fitness_new = np.array([self.fitness_with_constraint(ind) for ind in new_population])
            worst_idx = np.argmax(fitness_new)
            print(f"Mejor Fitness: {best_fitness}")
            print(f"Entropía reconstruida: {entropy_func(cv2.cvtColor(spline_cubic_transform(self.image, np.concatenate([[0.0], np.sort(np.clip(best_solution[:self.num_nodes], 0.05, 0.95)), [1.0]]), np.concatenate([[0.0], np.cumsum(np.abs(np.clip(best_solution[self.num_nodes:], 0, 1))) / np.sum(np.abs(np.clip(best_solution[self.num_nodes:], 0, 1))), [1.0]])), cv2.COLOR_BGR2GRAY)):.4f}")
            new_population[worst_idx] = best_solution.copy()
            population = np.array(new_population)
        x_inner = np.sort(np.clip(best_solution[:self.num_nodes], 0.05, 0.95))
        y_inner = np.clip(best_solution[self.num_nodes:], 0, 1)
        y_inner = np.cumsum(np.abs(y_inner))
        y_inner = y_inner / y_inner[-1]
        x_final = np.concatenate([[0.0], x_inner, [1.0]])
        y_final = np.concatenate([[0.0], y_inner, [1.0]])
        return x_final, y_final, best_fitness

# ========================== FUNCIÓN PRINCIPAL PARA LA APP ==========================

def process_image_and_entropy(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy_before = entropy_func(gray)
    ga = GeneticAlgorithm(image)
    x_nodes, y_nodes, _ = ga.evolve()
    enhanced_image = spline_cubic_transform(image, x_nodes, y_nodes)
    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    entropy_after = entropy_func(enhanced_gray)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    result_plot_path = os.path.join("results", f"{name}_plot.png")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path, enhanced_image)
    return result_path, entropy_before, entropy_after


import matplotlib.pyplot as plt

def process_image_and_plot(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy_before = entropy_func(gray)

    print(f"Entropía antes: {entropy_before}")
    ga = GeneticAlgorithm(image)
    x_nodes, y_nodes, _ = ga.evolve()
    enhanced_image = spline_cubic_transform(image, x_nodes, y_nodes)
    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    entropy_after = entropy_func(enhanced_gray)
    psnr = calculate_psnr(gray, enhanced_gray)

    # Crear plot comparativo
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].set_xlabel(f'Entropía: {entropy_before:.4f}')

    ax[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Imagen Mejorada (Spline)")
    ax[1].set_xlabel(
        f'x={np.round(x_nodes, 3)} \n'
        f'y={np.round(y_nodes, 3)} \n'
        f'Entropía: {entropy_after:.4f}, PSNR: {psnr:.2f} dB'
    )

    plt.tight_layout()
    
    result_plot_path = filepath.replace("uploads", "results").replace(".png", "_plot.png")
    os.makedirs(os.path.dirname(result_plot_path), exist_ok=True)
    plt.savefig(result_plot_path, dpi=300)
    plt.close()

    return result_plot_path, entropy_before, entropy_after
