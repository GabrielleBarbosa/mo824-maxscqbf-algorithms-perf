from typing import Tuple, List, Set
import numpy as np

def read_sc_max_qbf(path: str) -> Tuple[int, List[Set[int]], np.ndarray]:
    with open(path, 'r') as f:
        content = f.read().split()
        size = int(content[0])
        sets = [set() for _ in range(size)]
        A = np.zeros((size, size))
        content_index = 1
        sizes = []
        for i in range(size):
            sizes.append(int(content[content_index]))
            content_index += 1
        for i in range(size):
            for _ in range(sizes[i]):
                sets[i].add(int(content[content_index]))
                content_index += 1
        for i in range(size):
            for j in range(i, size):
                A[i, j] = float(content[content_index])
                content_index += 1
                A[j, i] = A[i, j]
    return size, A, sets