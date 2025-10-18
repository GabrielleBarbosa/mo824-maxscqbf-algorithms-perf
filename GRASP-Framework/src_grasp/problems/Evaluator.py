from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from src_grasp.solutions.Solution import Solution

E = TypeVar('E')

class Evaluator(Generic[E], ABC):
    @abstractmethod
    def get_domain_size(self) -> int:
        pass
    
    @abstractmethod
    def evaluate(self, sol: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_insertion_cost(self, elem: E, sol: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_removal_cost(self, elem: E, sol: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_exchange_cost(self, elem_in: E, elem_out: E, sol: Solution[E]) -> float:
        pass