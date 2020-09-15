# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np
import math

x_axis=[]
y_axis=[]
# Create the vectors X and Y
for i in range(-1000,1000,1):
    x = i/100
    y = (1/2) - (1/3.14)*math.atan(x)
    x_axis.append(x)
    y_axis.append(y)


# Create the plot
plt.plot(x_axis,y_axis)

# Add a title
plt.title('9b')

# Add X and y Label
plt.xlabel('x')
plt.ylabel('val')

# Add a grid
plt.grid(alpha=.1,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()