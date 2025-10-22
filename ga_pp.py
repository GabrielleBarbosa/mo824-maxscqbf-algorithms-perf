
import time
import csv
from ga.ga_scqbf import GA_SCQBF

def main():    
    output_file = "results/ga_pp.csv"
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
    
    print("Running GA computational experiments...")
    print("=" * 60)
    
    configs = [
        ("DEFAULT", False, True, True, 100, 0.05),
    ]
    
    for i in instances:
        print("\n" + "-" * 60)
        print(f"Instance: {i}")
        print("-" * 60)
        for config_name, enable_latin_hyper_cube, enable_mutate_or_crossover, enable_uniform_crossover, population_size, mutation_rate in configs:
            print(f"\nRunning {config_name}...")
            start_time = time.time()
            
            ga = GA_SCQBF(
                popSize=population_size,
                mutationRate=mutation_rate,
                filename=f"{parent_dir}/{i}.txt",
                enableLatinHyperCube=enable_latin_hyper_cube,
                enableMutateOrCrossover=enable_mutate_or_crossover,
                enableUniformCrossover=enable_uniform_crossover,
                timeLimit=30*60*1000, # in milliseconds
                targetValue=None,
                rng_seed=1,
            )
            
            best_sol, total_iterations, time_best_sol, iter_best_sol = ga.solve()
            end_time = time.time()

            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i,
                    config_name,
                    0, # seed is not directly configurable in the same way
                    total_iterations,
                    end_time - start_time,
                    best_sol.cost,
                    time_best_sol,
                    iter_best_sol,
                ])
            
            print(f"Cost: {best_sol.cost}, Size: {len(best_sol)}, Iterations: {total_iterations}, Time: {end_time - start_time:.3f}s")
            

if __name__ == "__main__":
    main()
