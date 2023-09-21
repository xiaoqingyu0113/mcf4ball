import pickle
import matplotlib.pyplot as plt



with open('figure.pkl', 'rb') as f:
    fig = pickle.load(f)

axes =  fig.axes

axes[0].text(0.48, -0.4, "(a)", transform=axes[0].transAxes, fontsize=18)
axes[0].text(0.48, -0.4, "(b)", transform=axes[0].transAxes, fontsize=18)

fig.savefig('tennis_9_i9.png')
# Display the loaded plot
plt.show()