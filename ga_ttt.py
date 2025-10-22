
import time
import csv
import multiprocessing
import os
from ga.ga_scqbf import GA_SCQBF

def worker(instance_name, target, config_name, enable_latin_hyper_cube, enable_mutate_or_crossover, enable_uniform_crossover, population_size, mutation_rate, process_index):
    output_file = f"results/ga_ttt_{process_index}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance",
            "config",
            "seed",
            "total_iterations",
            "total_time",
            "best_cost",
            "time_best_sol",
            "iterations_best_sol"
        ])

    parent_dir = "instances/sc_qbf"
    
    for r in range(50):
        start_time = time.time()
        
        ga = GA_SCQBF(
            popSize=population_size,
            mutationRate=mutation_rate,
            filename=f"{parent_dir}/{instance_name}.txt",
            enableLatinHyperCube=enable_latin_hyper_cube,
            enableMutateOrCrossover=enable_mutate_or_crossover,
            enableUniformCrossover=enable_uniform_crossover,
            timeLimit=10*60*1000, # in milliseconds
            targetValue=target
        )
        
        best_sol, total_iterations, time_best_sol, iter_best_sol = ga.solve()
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                instance_name,
                config_name,
                r,
                total_iterations,
                end_time - start_time,
                best_sol.cost,
                time_best_sol,
                iter_best_sol,
            ])

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    instances = [
        ("scqbf_025_1", 400),
        # ("scqbf_100_1", 400),
        # ("scqbf_200_1", 400),
        # ("scqbf_400_1", 400),
    ]
    
    configs = [
        ("DEFAULT", False, True, True, 100, 0.05),
    ]
    
    processes = []
    process_index = 0
    for i, target in instances:
        for config_name, enable_latin_hyper_cube, enable_mutate_or_crossover, enable_uniform_crossover, population_size, mutation_rate in configs:
            process = multiprocessing.Process(
                target=worker,
                args=(i, target, config_name, enable_latin_hyper_cube, enable_mutate_or_crossover, enable_uniform_crossover, population_size, mutation_rate, process_index)
            )
            processes.append(process)
            process.start()
            process_index += 1

    for process in processes:
        process.join()

    print("Computational experiments finished.")

if __name__ == "__main__":
    main()
