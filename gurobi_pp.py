import time
import csv
from gurobi import solver

def main():    
    output_file = "results/gurobi_pp.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance",
            "best_cost",
            "total_time",
            "optimal",
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
    
    for i in instances:
        print("\n" + "-" * 60)
        print(f"Instance: {i}")
        print("-" * 60)
        start_time = time.time()
        best_sol, optimal = solver.solve(f"{parent_dir}/{i}.txt", 30*60)
        end_time = time.time()

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i,
                best_sol,
                end_time - start_time,
                optimal,
            ])
        

if __name__ == "__main__":
    main()
