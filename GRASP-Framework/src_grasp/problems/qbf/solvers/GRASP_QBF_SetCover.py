import time
from typing import List
from src_grasp.metaheuristics.grasp.AbstractGRASP import AbstractGRASP
from src_grasp.problems.qbf.SetCoverQBF import SetCoverQBF
from src_grasp.solutions.Solution import Solution

class GRASP_QBF_SetCover(AbstractGRASP[int]):
    def __init__(self, alpha: float, iterations: int, filename: str,
                 time_limit: float = 1800.0, construction_type: str = "classic"):
        super().__init__(SetCoverQBF(filename), alpha, iterations, time_limit, construction_type)
        self.domain_size = self.ObjFunction.get_domain_size()

    def make_cl(self) -> List[int]:
        return list(range(self.domain_size))
    
    def make_rcl(self) -> List[int]:
        return []
    
    def update_cl(self) -> None:
        pass
    
    def create_empty_sol(self) -> Solution[int]:
        sol = Solution[int]()
        sol.cost = float('inf')
        return sol
    
    def constructive_heuristic(self) -> Solution[int]:
        self.CL = self.make_cl()
        self.RCL = self.make_rcl()
        self.sol = self.create_empty_sol()
        self.cost = float('inf')

        self.ensure_feasibility()

        while not self.constructive_stop_criteria():
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print("Time limit reached during constructive heuristic")
                break

            self.update_cl()
            if not self.CL:
                break

            candidate_costs = []
            for c in self.CL:
                cost = self.ObjFunction.evaluate_insertion_cost(c, self.sol)
                candidate_costs.append((c, cost))
            
            costs = [cost for _, cost in candidate_costs]
            min_cost = min(costs) if costs else 0
            max_cost = max(costs) if costs else 0

            if self.construction_type == "classic":
                threshold = min_cost + self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, cost in candidate_costs if cost <= threshold]
                if self.RCL:
                    chosen = self.rng.choice(self.RCL)
                else:
                    break

            elif self.construction_type == "random_greedy":
                if min_cost == max_cost:
                    chosen = self.rng.choice(self.CL)
                else:
                    weights = [1.0 / (cost - min_cost + 1e-6) for _, cost in candidate_costs]
                    # total = sum(weights)
                    chosen = self.rng.choices([c for c, _ in candidate_costs], weights=weights)[0]
            else:
                raise ValueError(f"Unknown construction type: {self.construction_type}")

            self.sol.append(chosen)
            self.CL.remove(chosen)

        self.cost = self.ObjFunction.evaluate(self.sol)
        return self.sol
    
    def ensure_feasibility(self) -> None:
        """Ensure the solution covers all variables"""
        uncovered = self.ObjFunction.get_uncovered_variables(self.sol)
        
        while uncovered:
            best_candidates = []
            best_coverage = -1
            
            for candidate in self.CL:
                new_coverage = len(uncovered & self.ObjFunction.sets[candidate])
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_candidates = [candidate]
                elif new_coverage == best_coverage:
                    best_candidates.append(candidate)
            
            if best_candidates:
                chosen = self.rng.choice(best_candidates)
                self.sol.append(chosen)
                self.CL.remove(chosen)
                uncovered = self.ObjFunction.get_uncovered_variables(self.sol)
            else:
                break

    def local_search(self) -> Solution[int]:
        """Local search that maintains feasibility"""
        improved = True
        
        while improved:
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print("Time limit reached during local search")
                break
                
            improved = False
            
            for cand_in in self.CL:
                for cand_out in list(self.sol):
                    if time.time() - self.start_time > self.time_limit:
                        break
                    
                    temp_sol = Solution(self.sol)
                    temp_sol.remove(cand_out)
                    temp_sol.append(cand_in)
                    
                    if self.ObjFunction.is_feasible(temp_sol):
                        delta_cost = self.ObjFunction.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                        if delta_cost < -1e-10:
                            self.sol.remove(cand_out)
                            self.sol.append(cand_in)
                            self.CL.remove(cand_in)
                            self.CL.append(cand_out)
                            self.ObjFunction.evaluate(self.sol)
                            improved = True
                            break
                if improved:
                    break
        
        return self.sol

    def constructive_stop_criteria(self) -> bool:
        """Stop when no more candidates"""
        return len(self.CL) == 0