##################### Image Creation #####################
import numpy as np
import pandas as pd
import math
from random import uniform
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

## Create the plot and save as image
sns.set(rc={'figure.figsize':(12,12)})
#sns.set(rc={'figure.figsize':(5,5)})

TRAIN_DATA_COUNT = 1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])# np.linspace(0, 2 * np.pi, 100)

##Generating x and y data
x = 16 * (np.sin(theta) ** 3)
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
#ax = sns.scatterplot(x=x, y=y)
ax = sns.scatterplot(x=x, y=y, markers=".")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

plt.savefig("./data/heart/heart_01.jpg", bbox_inches='tight')