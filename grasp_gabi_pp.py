import os
import sys
import time
import csv
import glob
import multiprocessing

from grasp_gabi.grasp_maxsc_qbf.algorithms.grasp_qbf_sc import GRASP_QBF_SC

def main():    
    output_file = "results/grasp_gabi_pp.csv"
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
        "scqbf_100_3",
        "scqbf_100_4",
        "scqbf_200_1",
        "scqbf_200_2",
        "scqbf_200_3",
        "scqbf_200_4",
        "scqbf_400_1",
        "scqbf_400_2",
        "scqbf_400_3",
    ]
    
    print("Running computational experiments...")
    print("=" * 60)
    
    for i in instances:
        print("\n" + "-" * 60)
        print(f"Instance: {i}")
        print("-" * 60)
        print(f"\nRunning GRASP_RANDOM_GREEDY...")
        start_time = time.time()
        
        solver = GRASP_QBF_SC(
            filename=f"{parent_dir}/{i}.txt",
            iterations=None,
            alpha=0.1,
            construction_method="random_plus_greedy",
            local_search_method="first_improving",
            time_limit=30*60
        )
        
        best_sol = solver.solve()
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i,
                "GRASP_RANDOM_GREEDY",
                1,
                solver.current_iter,
                end_time - start_time,
                best_sol.cost,
                solver.best_sol_time,
                solver.best_sol_iter,
            ])
        
        print(f"Cost: {best_sol.cost}, Size: {len(best_sol)}, Iterations: {solver.current_iter}, Time: {end_time - start_time:.3f}s")
            

if __name__ == "__main__":
    main()
