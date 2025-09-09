# Predictive Analysis with NumPy — Linear Regression from Scratch

# import numpy as np

# # 1) Data (NumPy arrays)
# hours = np.array([1, 2, 3, 4, 5], dtype=float)
# marks = np.array([30, 40, 50, 60, 70], dtype=float)

# # 2) Compute means
# x_mean = np.mean(hours)     # x̄
# y_mean = np.mean(marks)     # ȳ

# # 3) Compute slope (m) using OLS formula
# numerator = np.sum((hours - x_mean) * (marks - y_mean))     #xi - x̄
# denominator = np.sum((hours - x_mean)**2)                   # yi - ȳ
# m = numerator / denominator

# # 4) Compute intercept (c), where the line crosses the y-axis
# c = y_mean - m * x_mean

# # 5) Show the linear equation
# print(f"Fitted line: y = {m:.4f} * x + {c:.4f}")

# # 6) Predict for a new value (example: 6 hours)
# new_hours = 6.0
# predicted_marks = m * new_hours + c
# print(f"Predicted marks for {new_hours} hours: {predicted_marks:.4f}")

import numpy as np
import matplotlib.pyplot as plt

# Data (same as above)
hours = np.array([1, 2, 3, 4, 5], dtype=float)
marks = np.array([30, 40, 50, 60, 70], dtype=float)

# Fit (reuse formulas)
x_mean = np.mean(hours)
y_mean = np.mean(marks)
m = np.sum((hours - x_mean) * (marks - y_mean)) / np.sum((hours - x_mean)**2)
c = y_mean - m * x_mean

# Plot
plt.scatter(hours, marks)
# regression line for plotting: generate smooth x values
x_line = np.linspace(hours.min(), hours.max(), 100)
y_line = m * x_line + c
plt.plot(x_line, y_line)
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Linear Regression (NumPy)")
plt.grid(True)

# Save file
plt.savefig("regression_plot.png", dpi=150)
print("Saved regression_plot.png (current directory)")