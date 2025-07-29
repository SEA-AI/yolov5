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
import wandb 

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

def train_and_validate(gene_ranges, individual, device_id, return_dict, i):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from train import run as train

    import logging
    logging.getLogger().setLevel(logging.ERROR)

    with open("/home/gilsimas/git/yolov5/data/hyps/hyp.scratch-med.yaml", "r") as f:
        config = yaml.safe_load(f)
        for gene_name, gene_value in zip(gene_ranges.keys(), individual):

            if gene_value < gene_ranges[gene_name][0]:
                raise ValueError(f"Invalid value {gene_value} for gene {gene_name}. It should be between {gene_ranges[gene_name][0]} and {gene_ranges[gene_name][1]}")
            if gene_value > gene_ranges[gene_name][1]:
                raise ValueError(f"Invalid value {gene_value} for gene {gene_name}. It should be between {gene_ranges[gene_name][0]} and {gene_ranges[gene_name][1]}")
            
            config[gene_name] = gene_value
          
    with suppress_output():
        train_data = train(hyp=config, epochs=1, device=device_id)

    save_dir = train_data.save_dir

    with open(os.path.join(save_dir, "results.csv"), "r") as f:
        data = pd.read_csv(f)

    data["f1"] = 2 * data["   metrics/precision"] * data["      metrics/recall"] / (data["   metrics/precision"] + data["      metrics/recall"])
    
    max_map_epoch = data["metrics/mAP_0.5:0.95"].idxmax()

    f1 = data["f1"][max_map_epoch]
    map05_95 = data["metrics/mAP_0.5:0.95"][max_map_epoch]
    map_05 =   data["     metrics/mAP_0.5"][max_map_epoch]
    pr =       data["   metrics/precision"][max_map_epoch]
    rc =       data["      metrics/recall"][max_map_epoch]

    return_dict[i] = {"map05_95": map05_95, "map_05": map_05, "pr": pr, "rc": rc, "f1": f1}

    return return_dict[i]

class GeneticAlgorithm:
    def __init__(self,
                 cost_function=None,
                 pop_size=None,
                 mutation_rate=0.05,
                 crossover_rate=0.8,
                 elite_size=2,
                 tournament_size=3,
                 pop_per_gpu=2):

        self.cost_function = cost_function

        # Detect GPUs
        self.num_gpus = 2
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs detected!")
        
        self.generation_number = 0

        # Dynamic population size if not provided
        self.pop_size = pop_size or (self.num_gpus * pop_per_gpu)

        self.mutation_rate = max(0.01, min(0.5, mutation_rate))
        self.crossover_rate = max(0.5, min(1.0, crossover_rate))
        self.elite_size = max(2, min(5, elite_size))
        self.tournament_size = max(2, min(10, tournament_size))

        self.gene_ranges = {
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.9),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "im_compression_prob": (0.0, 1.0),
            "mosaic": (0.0, 1.0),  # image mosaic (probability)
            "mixup": (0.0, 0.5),  # image mixup (probability)
        }

        self.gene_length = len(self.gene_ranges)

        self.population = self._initialize_population()
        self.fitness_scores = [None] * self.pop_size

        print(f"Initialized GA with {self.pop_size} individuals across {self.num_gpus} GPUs")

    def _initialize_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]

    def generate_individual(self):
        genome = []
        for gene, ranges in self.gene_ranges.items():
            genome.append(random.uniform(ranges[0], ranges[1]))
        return genome

    def _evaluate_population(self):
        manager = mp.Manager()
        return_dict = manager.dict()
        num_gpus = self.num_gpus
        pop_size = self.pop_size

        for batch_start in range(0, pop_size, num_gpus):
            processes = []
            batch_end = min(batch_start + num_gpus, pop_size)

            for i, pop_idx in enumerate(range(batch_start, batch_end)):
                
                print(f"\n🧑 INDIVIDUAL {pop_idx} running in device {i}\n")

                individual = self.population[pop_idx]
                gpu_id = i  # because we cycle over num_gpus at each batch
                p = mp.Process(
                    target=train_and_validate,
                    args=(self.gene_ranges, individual, gpu_id, return_dict, pop_idx)
                )
                processes.append(p)
                p.start()
                time.sleep(20) # important otherwise both runs will try to read same .pt file

            for p in processes:
                p.join()

        self.fitness_scores = [return_dict[i] for i in range(pop_size)]

        for i, (individual, score) in enumerate(zip(self.population, self.fitness_scores)):
            run = wandb.init(
                project="genetic-algorithm-project",
                name=f"individual-{i}-generation-{self.generation_number}",
                reinit=True,  # Important for creating multiple runs in one script
                config={
                    "generation": self.generation_number,
                    "individual_id": i,
                    # Add any other hyperparameters you want
                }
            )

            # Log metrics for this individual
            wandb.log({
                "map05_95": score["map05_95"],
                "map_05": score["map_05"],
                "pr": score["pr"],
                "rc": score["rc"],
                "f1": score["f1"],
            })
            for gene_name, gene_value in zip(self.gene_ranges.keys(), individual):
                wandb.log({gene_name: gene_value})
            wandb.finish()

    def _get_elites(self):
        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.fitness_scores[i]["map05_95"], reverse=True)
        return [self.population[i] for i in sorted_indices[:self.elite_size]]

    def _tournament_selection(self):
        indices = random.sample(range(self.pop_size), self.tournament_size)
        best_index = max(indices, key=lambda i: self.fitness_scores[i]["map05_95"])
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
        best_index = max(range(self.pop_size), key=lambda i: self.fitness_scores[i]["map05_95"])
        return self.population[best_index]

    def evolve(self, generations=10):
        for gen in range(generations):
            print(f"\n🔁 Generation {gen+1}")
            self.generation_number = gen+1
            self._evaluate_population()
            best_score = max([fitness_score["map05_95"] for fitness_score in self.fitness_scores])
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
    elite_size=2,
    mutation_rate=0.1,
    crossover_rate=0.9,
    tournament_size=2,
    pop_per_gpu=2  # optional: population = num_gpus * 3
)

ga.evolve(generations=2)
print("Best individual:", ga.get_best_individual())
