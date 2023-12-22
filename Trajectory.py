import matplotlib.pyplot as plt
import numpy as np

# Trajectory
# Generate x and y values for the heart shape
t = np.linspace(0, 2*np.pi, 1000)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

# Calculate derivatives to obtain velocities
dx_dt = np.gradient(x, t)
dy_dt = np.gradient(y, t)

# Calculate derivatives again to obtain accelerations
d2x_dt2 = np.gradient(dx_dt, t)
d2y_dt2 = np.gradient(dy_dt, t)

# Plot the combined figure
plt.figure(figsize=(12, 10))

# Plot the acceleration trajectories
plt.subplot(3, 1, 1)
plt.plot(t, d2x_dt2, 'r', label='X-Acceleration')
plt.plot(t, d2y_dt2, 'b', label='Y-Acceleration')
plt.title('Acceleration vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

# Plot the velocity trajectories
plt.subplot(3, 1, 2)
plt.plot(t, dx_dt, 'r', label='X-Velocity')
plt.plot(t, dy_dt, 'b', label='Y-Velocity')
plt.title('Velocity vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)

# Plot the position trajectories
plt.subplot(3, 1, 3)
plt.plot(t, x, 'r', label='X-coordinate')
plt.plot(t, y, 'b', label='Y-coordinate')
plt.title('Position vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()