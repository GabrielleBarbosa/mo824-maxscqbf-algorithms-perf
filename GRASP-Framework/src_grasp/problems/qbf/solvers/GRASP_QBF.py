import time
from typing import List
from src_grasp.metaheuristics.grasp.AbstractGRASP import AbstractGRASP
from src_grasp.problems.qbf.QBF_Inverse import QBF_Inverse
from src_grasp.solutions.Solution import Solution

# class GRASP_QBF(AbstractGRASP[int]):
#     def __init__(self, alpha: float, iterations: int, filename: str, time_limit: float = 1800.0):
#         super().__init__(QBF_Inverse(filename), alpha, iterations, time_limit)

class GRASP_QBF(AbstractGRASP[int]):
    def __init__(self, alpha: float, iterations: int, filename: str,
                 time_limit: float = 1800.0, construction_type: str = "classic"):
        super().__init__(QBF_Inverse(filename), alpha, iterations, time_limit, construction_type)

    def make_cl(self) -> List[int]:
        return list(range(self.ObjFunction.get_domain_size()))
    
    def make_rcl(self) -> List[int]:
        return []
    
    def update_cl(self) -> None:
        pass
    
    def create_empty_sol(self) -> Solution[int]:
        sol = Solution[int]()
        sol.cost = 0.0
        return sol
    
    def local_search(self) -> Solution[int]:
        min_delta_cost = float('inf')
        best_cand_in = None
        best_cand_out = None
        
        while True:
            # Check time limit during local search
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print("Time limit reached during local search")
                break
                
            min_delta_cost = float('inf')
            self.update_cl()
            
            # Evaluate insertions
            for cand_in in self.CL:
                delta_cost = self.ObjFunction.evaluate_insertion_cost(cand_in, self.sol)
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = cand_in
                    best_cand_out = None
            
            # Evaluate removals
            for cand_out in self.sol:
                delta_cost = self.ObjFunction.evaluate_removal_cost(cand_out, self.sol)
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = None
                    best_cand_out = cand_out
            
            # Evaluate exchanges
            for cand_in in self.CL:
                for cand_out in self.sol:
                    # Check time limit during exchange evaluation
                    if time.time() - self.start_time > self.time_limit:
                        break
                    delta_cost = self.ObjFunction.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_cand_in = cand_in
                        best_cand_out = cand_out
            
            # Implement the best move if it reduces cost
            if min_delta_cost < -1e-10:
                if best_cand_out is not None:
                    self.sol.remove(best_cand_out)
                    self.CL.append(best_cand_out)
                if best_cand_in is not None:
                    self.sol.append(best_cand_in)
                    self.CL.remove(best_cand_in)
                self.ObjFunction.evaluate(self.sol)
            else:
                break
        
        return self.sol