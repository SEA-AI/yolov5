import random
import multiprocessing as mp
import math
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

def train_and_validate(gene_ranges, individual, device_id, return_dict, i, project_name, name, base_args):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        from train import run as train

        kwargs = dict()

        with open("/home/ubuntu/yolov5/data/hyps/hyp.sea-ai.yaml", "r") as f:
            config = yaml.safe_load(f)

        for gene_name, gene_value in zip(gene_ranges.keys(), individual):
            if (gene_ranges[gene_name][-1] == "log"):
                if gene_value < 10**(gene_ranges[gene_name][0]) or gene_value > 10**(gene_ranges[gene_name][1]):
                    print(gene_value, math.exp(gene_ranges[gene_name][0]), math.exp(gene_ranges[gene_name][1]))
                    raise ValueError(f"Invalid value {gene_value} for gene {gene_name}")
                gene_value = math.pow(10, gene_value)
            elif gene_ranges[gene_name][-1] == "linear":
                if gene_value < gene_ranges[gene_name][0] or gene_value > gene_ranges[gene_name][1]:
                    raise ValueError(f"Invalid value {gene_value} for gene {gene_name}")

            if gene_name in config:
                config[gene_name] = gene_value
            else:
                kwargs[gene_name] = gene_value

        # Merge base args
        kwargs.update(base_args)

        with suppress_output():
            train_data = train(hyp=config, 
                                device=device_id, 
                                project=project_name, 
                                name=name, 
                                data = "data/sea-ai-hyp-search.yaml",
                                # data = "data/coco128.yaml",
                                **kwargs)

        save_dir = train_data.save_dir
        with open(os.path.join(save_dir, "results.csv"), "r") as f:
            data = pd.read_csv(f)

        data["f1"] = 2 * data["   metrics/precision"] * data["      metrics/recall"] / (
            data["   metrics/precision"] + data["      metrics/recall"])

        max_map_epoch = data["metrics/mAP_0.05:0.95"].idxmax()

        return_dict[i] = {
            "map05_95": data["metrics/mAP_0.05:0.95"][max_map_epoch],
            "map_05": data["    metrics/mAP_0.05"][max_map_epoch],
            "pr": data["   metrics/precision"][max_map_epoch],
            "rc": data["      metrics/recall"][max_map_epoch],
            "f1": data["f1"][max_map_epoch],
        }

        print(f"✅ Process {i} finished successfully", flush=True)

    except Exception as e:
        print(f"❌ Exception in process {i} on device {device_id}: {e}", flush=True)
        return_dict[i] = {
            "map05_95": 0.0, "map_05": 0.0, "pr": 0.0, "rc": 0.0, "f1": 0.0
        }


class GeneticAlgorithm:
    def __init__(self,
                 pop_size=None,
                 mutation_rate=0.05,
                 crossover_rate=0.8,
                 elite_size=2,
                 tournament_size=3,
                 pop_per_gpu=2,
                 base_hyp = "/home/gilsimas/git/yolov5/data/hyps/hyp.sea-ai.yaml",
                 base_args = {
                     "imgsz": 1280,
                     "epochs": 20,
                     "weights": "yolov5s.pt",
                     "batch_size": -1,
                     "single_cls": False,
                     "multi_scale": False,
                     "close_mosaic": 5,
                     "single_cls_val": True,
                    }
                 ):

        with open(base_hyp, "r") as f:
            self.base_hyp = yaml.safe_load(f)
        
        self.base_args = base_args

        # Detect GPUs
        self.num_gpus = 8
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
            "lr0": (-5, -1, "float", "log"),
            "lrf": (0.1, 0.5, "float", "linear"),
            "momentum": (0.5, 1.0, "float", "linear"),
            "weight_decay": (-5, -2, "float", "log"),
            "warmup_momentum": (0.5, 1.0, "float", "linear"),
            "warmup_bias_lr": (0.0, 0.5, "float", "linear"),
            "hsv_h": (0.0, 0.1, "float", "linear"),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9, "float", "linear"),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9, "float", "linear"),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0, "float", "linear"),  # image rotation (+/- deg)
            "translate": (0.0, 0.9, "float", "linear"),  # image translation (+/- fraction)
            "scale": (0.0, 0.6, "float", "linear"),  # image scale (+/- gain)
            "shear": (0.0, 20.0, "float", "linear"),  # image shear (+/- deg)
            "perspective": (0.0, 0.001, "float", "linear"),  # image perspective (+/- fraction), range 0-0.001
            "compression": (0.0, 1.0, "float", "linear"),
            "mosaic": (0.0, 1.0, "float", "linear"),  # image mosaic (probability)
            "mixup": (0.0, 0.75, "float", "linear"),  # image mixup (probability)
            "pre_crop": (0.0, 1.0, "float", "linear"),
            "batch_size": (8, 16, "int", "linear"),
        }

        self.gene_length = len(self.gene_ranges)

        self.population = self._initialize_population()
        self.fitness_scores = [None] * self.pop_size

        print(f"Initialized GA with {self.pop_size} individuals across {self.num_gpus} GPUs")

    def _get_base_individual(self):
        genome = []
        for gene, ranges in self.gene_ranges.items():
            gene_value = random.uniform(ranges[0], ranges[1]) if gene not in self.base_hyp.keys() else self.base_hyp[gene]
            if ranges[2] == "int":
                gene_value = int(gene_value)
            genome.append(gene_value)
        return genome
            
    def _initialize_population(self):
        return [self._get_base_individual()]+[self.generate_individual() for _ in range(self.pop_size-1)]

    def generate_individual(self):
        genome = []
        for gene, ranges in self.gene_ranges.items():
            # print(gene, ranges)
            if ranges[-1] == "log":
                gene_value = random.uniform(ranges[0], ranges[1])
                gene_value = math.pow(10, gene_value)
                genome.append(gene_value)
            elif ranges[-1] == "linear":
                gene_value = random.uniform(ranges[0], ranges[1])
                if ranges[2] == "int":
                    gene_value = int(gene_value)
                genome.append(gene_value)
            else:
                # print(type(ranges), ranges, "wtf", ranges[-1], type(ranges[-1]), gene)
                raise ValueError(f"Invalid distribuition type: {ranges[-1]}")
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
                
                print(f"\n          🧑 INDIVIDUAL {pop_idx} running in device {i}\n")
                individual = self.population[pop_idx]
                gpu_id = i  # because we cycle over num_gpus at each batch
                p = mp.Process(
                    target=train_and_validate,
                    args=(self.gene_ranges, 
                          individual, 
                          gpu_id, 
                          return_dict, 
                          pop_idx, 
                          "genetic-algorithm-seaai-2",
                          f"individual-{pop_idx}-generation-{self.generation_number}",
                          self.base_args
                          )
                    )
                processes.append(p)
                p.start()
                time.sleep(20) # important otherwise both runs will try to read same .pt file

            for p in processes:
                p.join()

        print(f"✅ Return Dict KEYS:VALUES: {return_dict.keys()}:VALUES: {return_dict.values()}")
        self.fitness_scores = [return_dict[i] for i in range(pop_size)]

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


base_args = {
    "imgsz": 1280,
    "epochs": 25,
    "weights": "yolov5s.pt",
    "single_cls": False,
    "multi_scale": False,
    "close_mosaic": 5,
    "single_cls_val": True,
}

ga = GeneticAlgorithm(
    elite_size=2,
    mutation_rate=0.1,
    crossover_rate=0.9,
    tournament_size=2,
    pop_per_gpu=3,  # optional: population = num_gpus * 3
    base_hyp = "/home/ubuntu/yolov5/data/hyps/hyp.sea-ai.yaml",
    base_args = base_args
)

ga.evolve(generations=200)
print("Best individual:", ga.get_best_individual())


