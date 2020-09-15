# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np
import math

x_axis=[]
y_axis=[]
# Create the vectors X and Y
for i in range(-500,500,1):
    x = i/100
    y = 2 * math.exp(-abs(x)+abs((x-1)/2))
    x_axis.append(x)
    y_axis.append(y)


# Create the plot
plt.plot(x_axis,y_axis)

# Add a title
plt.title('2c')

# Add X and y Label
plt.xlabel('x')
plt.ylabel('ratio')

# Add a grid
plt.grid(alpha=.1,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()