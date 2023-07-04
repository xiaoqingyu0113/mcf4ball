import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mcf4ball.draw_util import draw_tennis_court


# Set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

draw_tennis_court(ax)



# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Rectangle')

# Display the plot
plt.show()
