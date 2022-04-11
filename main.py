import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

# Create space
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

# Create a function
x, y = np.meshgrid(x, y)
eq = (x**2 + y**2)**2

# Start point coordinates
x_p, y_p, z_p = 2, 3, 169
x_q, y_q, z_q = 1.4452998,   2.16794971, 46.08912857

# Draw plot
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, eq, alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlim3d(-2, 200)

# Draw start point
plt.plot(x_p, y_p, z_p, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
plt.plot(x_q, y_q, z_q, marker="o", markersize=20, markeredgecolor="green", markerfacecolor="red")

plt.show()


def func(x):  # Define a landscape function
    return (x[0]**2 + x[1]**2)**0.5  # x[0] here is 'x' and x[1] here is 'y'


# Define a start point
point = [2, 3]

direction = nd.Gradient(func)(point)
print("Direction:", direction)  # Here we got the right direction to move to the func min

step_size = 1

first_step = np.array(point) - step_size * direction
print("First step:", np.round(np.append(first_step, (first_step[0]**2 + first_step[1]**2)**2), 2))
# Here we got coordinate after the first move

# Now let's create a loop for subsequent moves to the minimum of the func
cur_p = np.array([2, 3])  # The algorithm starts from this point
l_rate = 0.01  # Learning rate (step size)
precision = 0.001  # This will stop the algorithm
max_iters = 1000  # Maximum number of iterations (steps)

prev_step_size = np.array([1, 1])
iter_counter = 0

while prev_step_size[0] > precision and iter_counter < max_iters:
    prev_p = cur_p  # Store current point value in the previous point value
    cur_p = cur_p - l_rate * nd.Gradient(func)(prev_p)  # Make steps (Gradient Descent)
    prev_step_size = abs(cur_p - prev_p)  # Change in point
    iter_counter = iter_counter + 1
    print("Iteration:", iter_counter, "\nCurrent step point:", cur_p)

print("The base of the mountain (min of the func) is at:", np.around(cur_p, decimals=3))
