import time
import csv
import multiprocessing

from grasp_gabi.grasp_maxsc_qbf.algorithms.grasp_qbf_sc import GRASP_QBF_SC

def worker(instance_name, target, process_index):
    output_file = f"results/grasp_gabi_ttt_{process_index}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance",
            "config",
            "seed",
            "target",
            "total_iterations",
            "total_time",
            "best_cost",
            "time_best_sol",
            "iterations_best_sol"
        ])

    parent_dir = "instances/sc_qbf"
    for r in range(50):
        start_time = time.time()
        solver = GRASP_QBF_SC(
            filename=f"{parent_dir}/{instance_name}.txt",
            iterations=None,
            alpha=0.1,
            construction_method="random_plus_greedy",
            local_search_method="first_improving",
            time_limit=10*60,
            seed=r,
            target=target
        )
        
        best_sol = solver.solve()
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                instance_name,
                "GRASP_RANDOM_GREEDY",
                r,
                target,
                solver.current_iter,
                end_time - start_time,
                best_sol.cost,
                solver.best_sol_time,
                solver.best_sol_iter,
            ])

def main():    
    instances = [
        ("scqbf_100_1", 16665 * 0.99),
        ("scqbf_200_1", 48906 * 0.99),
        ("scqbf_400_1", 321752 * 0.90),
        ("scqbf_400_1", 321752 * 0.95),
        ("scqbf_400_1", 321752 * 0.99),
    ]

    processes = []
    process_index = 0
    for i, target in instances:
        process = multiprocessing.Process(
            target=worker,
            args=(i, target, process_index)
        )
        processes.append(process)
        process.start()
        process_index += 1

    for process in processes:
        process.join()

    print("Computational experiments finished.")
    

if __name__ == "__main__":
    main()
