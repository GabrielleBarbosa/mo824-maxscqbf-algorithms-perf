import time
import csv
import multiprocessing
import os
from tabu.problems.sc_qbf.solvers.ts_sc_qbf import TS_SC_QBF

def worker(instance_name, target, config_name, tenure, local_search, strategy, process_index):
    output_file = f"results/tabu_ttt_{process_index}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance",
            "config",
            "seed",
            "target",
            "target_hit",
            "total_iterations",
            "total_time",
            "best_cost",
            "time_best_sol",
            "iterations_best_sol"
        ])

    parent_dir = "instances/sc_qbf"
    
    for r in range(50):
        start_time = time.time()
        
        ts = TS_SC_QBF(
            tenure=tenure, 
            filename=f"{parent_dir}/{instance_name}.txt",
            strategy=strategy,
            search_method=local_search,
            timeout=10*60, 
            random_seed=r, 
            target_value=-target,
            verbose=False
        )
        
        best_sol = ts.solve()
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                instance_name,
                config_name,
                r,
                target,
                -best_sol.cost >= target,
                ts.current_iter,
                end_time - start_time,
                -best_sol.cost,
                ts.best_sol_time,
                ts.best_sol_iter,
            ])

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    instances = [
        # ("scqbf_100_1", 16665 * 0.99),
        # ("scqbf_200_1", 48906 * 0.90),
        # ("scqbf_200_1", 48906 * 0.95),
        # ("scqbf_400_1", 141281 * 0.60),
        # ("scqbf_400_1", 141281 * 0.70),
        # ("scqbf_400_1", 141281 * 0.80),
        # ("scqbf_400_1", 141281 * 0.90),
        # ("scqbf_400_1", 141281 * 0.95),
        ("scqbf_100_3", 20471 * 0.99),
        ("scqbf_200_3", 50470 * 0.90),
        ("scqbf_200_3", 50470 * 0.95),
        ("scqbf_200_3", 50470 * 0.99),
        ("scqbf_400_3", 140016 * 0.50),
        ("scqbf_400_3", 140016 * 0.80),
        ("scqbf_400_3", 140016 * 0.99)
    ]
    
    configs = [
        ("BEST_PROBABILISTIC", 0.2, "best_improving", "probabilistic"),
    ]
    
    processes = []
    process_index = 0
    for i, target in instances:
        for config_name, tenure, local_search, strategy in configs:
            process = multiprocessing.Process(
                target=worker,
                args=(i, target, config_name, tenure, local_search, strategy, process_index)
            )
            processes.append(process)
            process.start()
            process_index += 1

    for process in processes:
        process.join()

    print("Computational experiments finished.")

if __name__ == "__main__":
    main()
