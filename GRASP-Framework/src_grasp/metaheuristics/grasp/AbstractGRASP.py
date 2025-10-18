import random
import time
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Optional
from src_grasp.problems.Evaluator import Evaluator
from src_grasp.solutions.Solution import Solution

E = TypeVar('E')

class AbstractGRASP(Generic[E], ABC):
    verbose = True
    rng = random.Random(0)
    
    # def __init__(self, obj_function: Evaluator[E], alpha: float, iterations: int, time_limit: float = 1800.0):
    #     self.ObjFunction = obj_function
    #     self.alpha = alpha
    #     self.iterations = iterations
    #     self.time_limit = time_limit  # 30 minutes in seconds
    #     self.best_cost: float = float('inf')
    #     self.cost: float = float('inf')
    #     self.best_sol: Optional[Solution[E]] = None
    #     self.sol: Optional[Solution[E]] = None
    #     self.CL: Optional[List[E]] = None
    #     self.RCL: Optional[List[E]] = None
    #     self.start_time: float = 0.0
    def __init__(self, obj_function: Evaluator[E], alpha: float, iterations: int,
             time_limit: float = 1800.0, construction_type: str = "classic"):
        self.ObjFunction = obj_function
        self.alpha = alpha
        self.iterations = iterations
        self.time_limit = time_limit
        self.construction_type = construction_type  # new
        self.best_cost: float = float('inf')
        self.cost: float = float('inf')
        self.best_sol: Optional[Solution[E]] = None
        self.sol: Optional[Solution[E]] = None
        self.CL: Optional[List[E]] = None
        self.RCL: Optional[List[E]] = None
        self.start_time: float = 0.0

    
    @abstractmethod
    def make_cl(self) -> List[E]:
        pass
    
    @abstractmethod
    def make_rcl(self) -> List[E]:
        pass
    
    @abstractmethod
    def update_cl(self) -> None:
        pass
    
    @abstractmethod
    def create_empty_sol(self) -> Solution[E]:
        pass
    
    @abstractmethod
    def local_search(self) -> Solution[E]:
        pass
    
    # def constructive_heuristic(self) -> Solution[E]:
    #     self.CL = self.make_cl()
    #     self.RCL = self.make_rcl()
    #     self.sol = self.create_empty_sol()
    #     self.cost = float('inf')
        
    #     while not self.constructive_stop_criteria():
    #         # Check time limit
    #         if time.time() - self.start_time > self.time_limit:
    #             if self.verbose:
    #                 print("Time limit reached during constructive heuristic")
    #             break
                
    #         max_cost = float('-inf')
    #         min_cost = float('inf')
    #         self.cost = self.ObjFunction.evaluate(self.sol)
    #         self.update_cl()
            
    #         for c in self.CL:
    #             delta_cost = self.ObjFunction.evaluate_insertion_cost(c, self.sol)
    #             if delta_cost < min_cost:
    #                 min_cost = delta_cost
    #             if delta_cost > max_cost:
    #                 max_cost = delta_cost
            
    #         for c in self.CL:
    #             delta_cost = self.ObjFunction.evaluate_insertion_cost(c, self.sol)
    #             if delta_cost <= min_cost + self.alpha * (max_cost - min_cost):
    #                 self.RCL.append(c)
            
    #         rnd_index = self.rng.randint(0, len(self.RCL) - 1)
    #         in_cand = self.RCL[rnd_index]
    #         self.CL.remove(in_cand)
    #         self.sol.append(in_cand)
    #         self.ObjFunction.evaluate(self.sol)
    #         self.RCL.clear()
        
    #     return self.sol
    
    def constructive_heuristic(self) -> Solution[E]:
        self.CL = self.make_cl()
        self.RCL = self.make_rcl()
        self.sol = self.create_empty_sol()
        self.cost = float('inf')

        while not self.constructive_stop_criteria():
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print("Time limit reached during constructive heuristic")
                break

            self.update_cl()
            if not self.CL:
                break

            # Evaluate insertion costs for each candidate
            candidate_costs = [(c, self.ObjFunction.evaluate_insertion_cost(c, self.sol))
                            for c in self.CL]
            costs = [cost for _, cost in candidate_costs]
            min_cost, max_cost = min(costs), max(costs)

            if self.construction_type == "classic":
                # Classic GRASP RCL
                threshold = min_cost + self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, cost in candidate_costs if cost <= threshold]
                chosen = self.rng.choice(self.RCL)

            elif self.construction_type == "random_greedy":
                # Probabilistic selection based on inverse cost (lower = better)
                weights = [1.0 / (cost - min_cost + 1e-6) for _, cost in candidate_costs]
                total = sum(weights)
                probs = [w / total for w in weights]
                chosen = self.rng.choices([c for c, _ in candidate_costs], probs)[0]

            else:
                raise ValueError(f"Unknown construction type: {self.construction_type}")

            # Apply insertion
            # self.sol.add(chosen)
            self.sol.append(chosen)
            self.CL.remove(chosen)

        self.cost = self.ObjFunction.evaluate(self.sol)
        return self.sol

    def solve(self) -> Solution[E]:
        self.start_time = time.time()
        self.best_sol = self.create_empty_sol()
        
        for i in range(self.iterations):
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print(f"Time limit reached after {i} iterations")
                break
                
            self.constructive_heuristic()
            
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print(f"Time limit reached before local search in iteration {i}")
                break
                
            self.local_search()
            
            if self.best_sol.cost > self.sol.cost:
                self.best_sol = Solution(self.sol)
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")
        
        total_time = time.time() - self.start_time
        if self.verbose:
            print(f"Total execution time: {total_time:.2f} seconds")
        
        return self.best_sol
    
    def constructive_stop_criteria(self) -> bool:
        return not (self.cost > self.sol.cost)