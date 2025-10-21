import os
import csv
import time
import random
from src_grasp.problems.qbf.solvers.GRASP_QBF import GRASP_QBF
from src_grasp.problems.qbf.solvers.GRASP_QBF_SetCover import GRASP_QBF_SetCover

instances = [
    "instances_grasp/qbf/qbf040",

]

output_file = "results_grasp_qbf.csv"
num_runs = 2           # Number of independent runs per instance
alpha = 0.3             # Greediness-randomness parameter
iterations = 1000       # GRASP iterations
time_limit = 600        # seconds per run
construction_type = "random_greedy"  # or "classic"


os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "instance",
        "run_id",
        "seed",
        "alpha",
        "iterations",
        "construction_type",
        "best_cost",
        "target_value",
        "cpu_time_to_target",
        "total_time",
        "success_flag"
    ])



for instance in instances:
    print(f"\n=== Running instance {instance} ===")
    target_value = None

    for run_id in range(1, num_runs + 1):
        seed = random.randint(0, 999999)
        random.seed(seed)

        grasp = GRASP_QBF_SetCover(
            alpha=alpha,
            iterations=iterations,
            filename=instance,
            time_limit=time_limit,
            construction_type=construction_type
        )

        start_time = time.time()
        best_sol = grasp.solve()  # assumes your AbstractGRASP implements .solve()
        total_time = time.time() - start_time

        best_cost = best_sol.cost
        if target_value is None:
            target_value = best_cost

        cpu_time_to_target = getattr(grasp, "time_to_target", total_time)
        success_flag = 1 if best_cost <= target_value else 0

        # Save results
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(instance),
                run_id,
                seed,
                alpha,
                iterations,
                construction_type,
                best_cost,
                target_value,
                cpu_time_to_target,
                total_time,
                success_flag
            ])

        print(f"Run {run_id}/{num_runs}: best = {best_cost:.2f}, time = {total_time:.2f}s")
