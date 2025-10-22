
import time
import csv
import multiprocessing
import os
import numpy as np
from typing import List, Set

from grasp.grasp_scmax import GRASP_SC_MAX_QBF, GRASPConfig
from grasp.sc_model import SCMaxQBF
from grasp.qbf import read_sc_max_qbf


def worker(instance_name, target, config_name, alpha, ls_mode, lambda_balance, process_index):
    output_file = f"results/grasp_ttt_{process_index}.csv"
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
        
        _, Q, sets = read_sc_max_qbf(f"{parent_dir}/{instance_name}.txt")
        model = SCMaxQBF(Q, sets)
        
        cfg = GRASPConfig(
            alpha=alpha,
            ls_mode=ls_mode,
            lambda_balance=lambda_balance,
            time_limit=10*60,
            seed=r,
            target_value=target
        )
        
        grasp = GRASP_SC_MAX_QBF(model, cfg)
        
        best_S, best_val, ttt, total_time, total_iterations, time_best_sol, iter_best_sol = grasp.run()
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                instance_name,
                config_name,
                r,
                total_iterations,
                end_time - start_time,
                best_val,
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
        ("DEFAULT", 0.3, "best", 0.4),
    ]
    
    processes = []
    process_index = 0
    for i, target in instances:
        for config_name, alpha, ls_mode, lambda_balance in configs:
            process = multiprocessing.Process(
                target=worker,
                args=(i, target, config_name, alpha, ls_mode, lambda_balance, process_index)
            )
            processes.append(process)
            process.start()
            process_index += 1

    for process in processes:
        process.join()

    print("Computational experiments finished.")

if __name__ == "__main__":
    main()
