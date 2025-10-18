from typing import List
from src_grasp.problems.Evaluator import Evaluator
from src_grasp.solutions.Solution import Solution

class QBF(Evaluator[int]):
    def __init__(self, filename: str):
        self.size, self.A = self.read_input(filename)
        self.variables = self.allocate_variables()
    
    def set_variables(self, sol: Solution[int]) -> None:
        self.reset_variables()
        for elem in sol:
            self.variables[elem] = 1.0
    
    def get_domain_size(self) -> int:
        return self.size
    
    def evaluate(self, sol: Solution[int]) -> float:
        self.set_variables(sol)
        sol.cost = self.evaluate_qbf()
        return sol.cost
    
    def evaluate_qbf(self) -> float:
        aux = 0.0
        total_sum = 0.0
        vec_aux = [0.0] * self.size
        
        for i in range(self.size):
            for j in range(self.size):
                aux += self.variables[j] * self.A[i][j]
            vec_aux[i] = aux
            total_sum += aux * self.variables[i]
            aux = 0.0
        
        return total_sum
    
    def evaluate_insertion_cost(self, elem: int, sol: Solution[int]) -> float:
        self.set_variables(sol)
        return self.evaluate_insertion_qbf(elem)
    
    def evaluate_insertion_qbf(self, i: int) -> float:
        if self.variables[i] == 1:
            return 0.0
        return self.evaluate_contribution_qbf(i)
    
    def evaluate_removal_cost(self, elem: int, sol: Solution[int]) -> float:
        self.set_variables(sol)
        return self.evaluate_removal_qbf(elem)
    
    def evaluate_removal_qbf(self, i: int) -> float:
        if self.variables[i] == 0:
            return 0.0
        return -self.evaluate_contribution_qbf(i)
    
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution[int]) -> float:
        self.set_variables(sol)
        return self.evaluate_exchange_qbf(elem_in, elem_out)
    
    def evaluate_exchange_qbf(self, in_idx: int, out_idx: int) -> float:
        if in_idx == out_idx:
            return 0.0
        if self.variables[in_idx] == 1:
            return self.evaluate_removal_qbf(out_idx)
        if self.variables[out_idx] == 0:
            return self.evaluate_insertion_qbf(in_idx)
        
        total_sum = self.evaluate_contribution_qbf(in_idx)
        total_sum -= self.evaluate_contribution_qbf(out_idx)
        total_sum -= (self.A[in_idx][out_idx] + self.A[out_idx][in_idx])
        
        return total_sum
    
    def evaluate_contribution_qbf(self, i: int) -> float:
        total_sum = 0.0
        for j in range(self.size):
            if i != j:
                total_sum += self.variables[j] * (self.A[i][j] + self.A[j][i])
        total_sum += self.A[i][i]
        return total_sum
    
    def read_input(self, filename: str):
        with open(filename, 'r') as file:
            lines = file.readlines()
            size = int(lines[0].strip())
            
            A = [[0.0] * size for _ in range(size)]
            line_idx = 1
            
            for i in range(size):
                values = list(map(float, lines[line_idx].split()))
                line_idx += 1
                for j in range(i, size):
                    A[i][j] = values[j - i]
                    if j > i:
                        A[j][i] = 0.0
            
            return size, A
    
    def allocate_variables(self) -> List[float]:
        return [0.0] * self.size
    
    def reset_variables(self) -> None:
        for i in range(self.size):
            self.variables[i] = 0.0
    
    def print_matrix(self) -> None:
        for i in range(self.size):
            for j in range(i, self.size):
                print(f"{self.A[i][j]} ", end="")
            print()