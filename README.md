# Transportation-problem
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import seaborn as sns
import matplotlib.pyplot as plt
# difine supply and demand
supply = [20, 30, 25]
demand = [10, 25, 20, 20]
#  Define the cost matrix (source x destination)
cost = np.array([
    [8, 6, 10, 9],
    [9, 12, 13, 7],
    [14, 9, 16, 5]
])
#  Problem Setup
num_sources = len(supply)
num_destinations = len(demand)

model = LpProblem("Transportation_Problem", LpMinimize)

# 4. Decision Variables
x = LpVariable.dicts("x", (range(num_sources), range(num_destinations)), lowBound=0, cat='Continuous')

# 5. Objective Function
model += lpSum(cost[i][j] * x[i][j] for i in range(num_sources) for j in range(num_destinations))

# 6. Constraints
# Supply constraints
for i in range(num_sources):
    model += lpSum(x[i][j] for j in range(num_destinations)) <= supply[i]

# Demand constraints
for j in range(num_destinations):
    model += lpSum(x[i][j] for i in range(num_sources)) >= demand[j]

# 7. Solve the model
model.solve()

# 8. Output the results
result_matrix = np.zeros((num_sources, num_destinations))
for i in range(num_sources):
    for j in range(num_destinations):
        result_matrix[i][j] = x[i][j].varValue

df_result = pd.DataFrame(result_matrix, 
                         index=[f"Source {i+1}" for i in range(num_sources)],
                         columns=[f"Destination {j+1}" for j in range(num_destinations)])

print("Status:", LpStatus[model.status])
print("Total Minimum Cost:", value(model.objective))
print("\nOptimal Transport Plan:\n")
print(df_result)

# 9. Heatmap Visualization
plt.figure(figsize=(8, 5))
sns.heatmap(df_result, annot=True, cmap="YlGnBu", fmt=".0f")
plt.title("Optimal Transportation Plan (Units)")
plt.show()
