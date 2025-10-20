import gurobipy as gp
from gurobipy import GRB

def _read_input(filename: str):
        with open(filename, 'r') as f:
            content = f.read().split()
            size = int(content[0])
            S = [[] for _ in range(size)]
            A = [[0 for _ in range(size)] for _ in range(size)]
            content_index = 1
            sizes = []
            for i in range(size):
                sizes.append(int(content[content_index]))
                content_index += 1
            for i in range(size):
                for _ in range(sizes[i]):
                    S[i].append(int(content[content_index]))
                    content_index += 1
            for i in range(size):
                for j in range(i, size):
                    A[i][j] = float(content[content_index])
                    content_index += 1
                    if i != j:
                        A[j][i] = 0.0
        return size, S, A
        
def solve(instance_file: str, timeout: int):
    n, s, A = _read_input(instance_file)

    model = gp.Model("max-sc-qbf")   
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    y = model.addVars(pairs, vtype=GRB.BINARY, name="y")

    model.setObjective(
        sum(A[i][j] * y[i,j] for i in range(n) for j in range(i, n)),
        GRB.MAXIMIZE
    )

    for i in range(n):
        for j in range(i, n):
            model.addConstr(y[i,j] <= x[i])
            model.addConstr(y[i,j] <= x[j])
            model.addConstr(y[i,j] >= x[i] + x[j] - 1)

    for k in range(1, n+1):
        indexes = []
        for i in range(n):
            if k in s[i]:
                indexes.append(i)
        model.addConstr(gp.quicksum(x[i] for i in indexes) >= 1)

    model.setParam('TimeLimit', timeout)
    model.optimize()
    
    return model.ObjVal, model.status == GRB.OPTIMAL