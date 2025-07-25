import random
import multiprocessing as mp
import yaml
import pandas as pd
import os
import time
import sys
import os
import contextlib
from contextlib import redirect_stdout, redirect_stderr

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def train_and_validate(individual, device_id, return_dict, i):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from train import run as train

    import logging
    logging.getLogger().setLevel(logging.ERROR)

    with open("/home/gilsimas/git/yolov5/data/hyps/hyp.scratch-med.yaml", "r") as f:
        config = yaml.safe_load(f)
        config["cls_pw"] = individual[0]
          
    with suppress_output():
        train_data = train(hyp=config, epochs=1, device=device_id)

    save_dir = train_data.save_dir

    with open(os.path.join(save_dir, "results.csv"), "r") as f:
        data = pd.read_csv(f)

    data["f1"] = 2 * data["   metrics/precision"] * data["      metrics/recall"] / (data["   metrics/precision"] + data["      metrics/recall"])
    max_f1 = data["f1"].max()

    return_dict[i] = max_f1

    return max_f1

class GeneticAlgorithm:
    def __init__(self,
                 gene_length,
                 cost_function=None,
                 pop_size=None,
                 mutation_rate=0.05,
                 crossover_rate=0.8,
                 elite_size=2,
                 tournament_size=3,
                 pop_per_gpu=2):

        self.gene_length = gene_length
        self.cost_function = cost_function

        # Detect GPUs
        self.num_gpus = 2
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs detected!")

        # Dynamic population size if not provided
        self.pop_size = pop_size or (self.num_gpus * pop_per_gpu)

        self.mutation_rate = max(0.01, min(0.5, mutation_rate))
        self.crossover_rate = max(0.5, min(1.0, crossover_rate))
        self.elite_size = max(2, min(5, elite_size))
        self.tournament_size = max(2, min(10, tournament_size))

        self.population = self._initialize_population()
        self.fitness_scores = [None] * self.pop_size

        print(f"Initialized GA with {self.pop_size} individuals across {self.num_gpus} GPUs")

    def _initialize_population(self):
        return [[random.random() for _ in range(self.gene_length)]
                for _ in range(self.pop_size)]

    def _evaluate_population(self):
        manager = mp.Manager()
        return_dict = manager.dict()
        num_gpus = self.num_gpus
        pop_size = self.pop_size

        for batch_start in range(0, pop_size, num_gpus):
            processes = []
            batch_end = min(batch_start + num_gpus, pop_size)

            for i, pop_idx in enumerate(range(batch_start, batch_end)):
                
                print(f"\n🧑 INDIVIDUAL {pop_idx} \n")

                individual = self.population[pop_idx]
                gpu_id = i  # because we cycle over num_gpus at each batch
                p = mp.Process(
                    target=train_and_validate,
                    args=(individual, gpu_id, return_dict, pop_idx)
                )
                processes.append(p)
                p.start()
                time.sleep(5)

            for p in processes:
                p.join()

        self.fitness_scores = [return_dict[i] for i in range(pop_size)]

    def _get_elites(self):
        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitness_scores[i])
        return [self.population[i] for i in sorted_indices[:self.elite_size]]

    def _tournament_selection(self):
        indices = random.sample(range(self.pop_size), self.tournament_size)
        best_index = min(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_index]

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:]
        point = random.randint(1, self.gene_length - 1)
        return parent1[:point] + parent2[point:]

    def _mutate(self, individual):
        return [gene if random.random() > self.mutation_rate else random.random()
                for gene in individual]

    def get_best_individual(self):
        best_index = min(range(self.pop_size), key=lambda i: self.fitness_scores[i])
        return self.population[best_index]

    def evolve(self, generations=10):
        for gen in range(generations):
            print(f"\n🔁 Generation {gen+1}")
            self._evaluate_population()
            best_score = min(self.fitness_scores)
            print(f"✅ Best Fitness: {best_score:.4f}")

            new_population = self._get_elites()

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population

ga = GeneticAlgorithm(
    gene_length=2,
    elite_size=2,
    mutation_rate=0.1,
    crossover_rate=0.9,
    tournament_size=2,
    pop_per_gpu=2  # optional: population = num_gpus * 3
)

ga.evolve(generations=2)
print("Best individual:", ga.get_best_individual())
