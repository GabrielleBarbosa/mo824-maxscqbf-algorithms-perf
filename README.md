# mo824-ativ5-max-sc-qbf-algorithms-profiles

This is a repository destined to run experiments on the problem MAX-SC-QBF on different methods to then use the results to compute Performance Profile and TTT-Plot metrics.

The methods implemented are:
- Exact PLI Solver (Gurobi)
- GRASP with random plus greedy strategy
- Tabu Search with probabilistic strategy
- GA with mutate or crossover and uniform crossover

The instances used in the experiments are in the directory "instances" and have their size in the name


### Performance profile
To the performance profile, all instances are ran once with a limit of 30 minutes. There is a script for each method with the suffix "_pp" and generate csv files in the directory "results", with the same name as the script file.

how to run:
```
python3 gurobi_pp.py
python3 grasp_pp.py
python3 tabu_pp.py
python3 ga_pp.py
```

example result:

./results/grasp_pp.csv
```
instance,config,seed,total_iterations,total_time,best_cost,time_best_sol,iterations_best_sol
scqbf_025_1,DEFAULT,0,34535,1800.0239126682281,1316.0,0.886,14
scqbf_025_2,DEFAULT,0,33356,1800.040390253067,1279.0,0.367,5
```


### TTT Plots
To the TTT-Plot, some instances are selected with a target value, wich are specified on the top of the files, and are ran 50 times with different random seeds and a limit of 10 minutes per run, stopping early if the target value is reached. There is also a script for each method, but with the suffix "_ttt", and it parallelizes each instance target in a process, saving multiple csv files on the "results" directory, with the same name of the script but concatenated with the instance index. 

how to run:

```
python3 grasp_ttt.py
python3 tabu_ttt.py
python3 ga_ttt.py
```

example result:

./results/grasp_ttt_0.csv
```
instance,config,seed,target,target_hit,total_iterations,total_time,best_cost,time_best_sol,iterations_best_sol
scqbf_100_3,GRASP_RANDOM_GREEDY,0,20266.29,True,1,0.5019333362579346,20272.0,0.49961090087890625,1
scqbf_100_3,GRASP_RANDOM_GREEDY,1,20266.29,True,1,0.6238758563995361,20420.0,0.6210429668426514,1
```