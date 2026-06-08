import numpy as np

data = np.loadtxt('./arm_samples.csv', delimiter=',', skiprows=1)
# columns: time, x, y, z
x = data[:, 1]
y = data[:, 2]

print(f"var_x  = {np.var(x):.2e}  (σ_x = {np.std(x)*1000:.2f} mm)")
print(f"var_y  = {np.var(y):.2e}  (σ_y = {np.std(y)*1000:.2f} mm)")
print(f"cov_xy = {np.cov(x, y)[0,1]:.2e}")
print(f"샘플 수: {len(x)}")