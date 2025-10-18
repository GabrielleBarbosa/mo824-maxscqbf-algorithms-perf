from typing import TypeVar, Generic, Optional

E = TypeVar('E')

class Solution(Generic[E], list):
    def __init__(self, other: Optional['Solution[E]'] = None):
        if other is None:
            super().__init__()
            self.cost = float('inf')
        else:
            super().__init__(other)
            self.cost = other.cost
    
    def __str__(self) -> str:
        return f"Solution: cost=[{self.cost}], size=[{len(self)}], elements={super().__str__()}"