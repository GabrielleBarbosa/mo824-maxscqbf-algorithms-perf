
import time
import csv
import numpy as np
from typing import List, Set

from grasp.grasp_scmax import GRASP_SC_MAX_QBF, GRASPConfig
from grasp.sc_model import SCMaxQBF
from grasp.qbf import read_sc_max_qbf

def main():    
    output_file = "results/grasp_pp.csv"
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
    instances = [
        "scqbf_025_1",
        "scqbf_025_2",
        "scqbf_050_1",
        "scqbf_050_2",
        "scqbf_100_1",
        "scqbf_100_2",
        "scqbf_200_1",
        "scqbf_200_2",
        "scqbf_400_1",
        "scqbf_400_2",
    ]
    
    print("Running GRASP computational experiments...")
    print("=" * 60)
    
    configs = [
        ("DEFAULT", 0.3, "best", 0.4),
    ]
    
    for i in instances:
        print("\n" + "-" * 60)
        print(f"Instance: {i}")
        print("-" * 60)
        for config_name, alpha, ls_mode, lambda_balance in configs:
            print(f"\nRunning {config_name}...")
            start_time = time.time()
            
            _, Q, sets = read_sc_max_qbf(f"{parent_dir}/{i}.txt")
            model = SCMaxQBF(Q, sets)
            
            cfg = GRASPConfig(
                alpha=alpha,
                ls_mode=ls_mode,
                lambda_balance=lambda_balance,
                time_limit=30*60,
                seed=1
            )
            
            grasp = GRASP_SC_MAX_QBF(model, cfg)
            
            best_S, best_val, ttt, total_time, total_iterations, time_best_sol, iter_best_sol = grasp.run()
            end_time = time.time()

            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i,
                    config_name,
                    1,
                    total_iterations,
                    end_time - start_time,
                    best_val,
                    time_best_sol,
                    iter_best_sol,
                ])
            
            print(f"Cost: {best_val}, Size: {len(best_S)}, Iterations: {total_iterations}, Time: {end_time - start_time:.3f}s")
            

if __name__ == "__main__":
    main()
