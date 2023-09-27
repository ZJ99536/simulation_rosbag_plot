import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def sinplot(flip = 2):
    x = np.linspace(0,20, 50)
    for i in range(1,5):
        plt.plot(x, np.cos(x + i * 0.8) * (9 - 2*i) * flip)

sinplot()
plt.show()
