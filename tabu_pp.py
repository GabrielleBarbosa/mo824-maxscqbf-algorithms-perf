import time
import csv
from tabu.problems.sc_qbf.solvers.ts_sc_qbf import TS_SC_QBF


def main():    
    output_file = "results/tabu_pp.csv"
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
        "scqbf_050_1",
        "scqbf_100_1",
        "scqbf_200_1",
        "scqbf_400_1",
    ]
    
    print("Running computational experiments...")
    print("=" * 60)
    
    configs = [
        ("BEST_PROBABILISTIC", 0.2, "best_improving", "probabilistic"),
    ]
    
    for i in instances:
        print("\n" + "-" * 60)
        print(f"Instance: {i}")
        print("-" * 60)
        for config_name, tenure, local_search, strategy in configs:
            print(f"\nRunning {config_name}...")
            start_time = time.time()
            
            ts = TS_SC_QBF(
                tenure=tenure, 
                iterations=10000, 
                filename=f"{parent_dir}/{i}.txt",
                strategy=strategy,
                search_method=local_search,
                timeout=30*60, 
                random_seed=1, 
                # target_value=-735.0,
            )
            
            best_sol = ts.solve()
            end_time = time.time()

            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i,
                    config_name,
                    1,
                    ts.current_iter,
                    end_time - start_time,
                    -best_sol.cost,
                    ts.best_sol_time,
                    ts.best_sol_iter,
                ])
            
            print(f"Cost: {-best_sol.cost}, Size: {len(best_sol)}, Iterations: {ts.current_iter}, Time: {end_time - start_time:.3f}s")
            

if __name__ == "__main__":
    main()