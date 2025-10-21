from typing import List, Set
from src_grasp.problems.qbf.QBF_Inverse import QBF_Inverse
from src_grasp.solutions.Solution import Solution

class SetCoverQBF(QBF_Inverse):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.sets = self.create_default_sets()
        self.universe = set(range(self.size))
    
    def create_default_sets(self) -> List[Set[int]]:
        """Create default sets where each set covers only itself"""
        sets = []
        for i in range(self.size):
            sets.append({i})
        return sets
    
    def is_feasible(self, sol: Solution[int]) -> bool:
        """Check if solution covers all variables"""
        covered = set()
        for set_idx in sol:
            covered |= self.sets[set_idx]
        return covered == self.universe
    
    def get_uncovered_variables(self, sol: Solution[int]) -> Set[int]:
        """Get variables not covered by current solution"""
        covered = set()
        for set_idx in sol:
            covered |= self.sets[set_idx]
        return self.universe - covered