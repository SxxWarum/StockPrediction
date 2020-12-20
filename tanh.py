import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2.5, 2.5, 1000)

y = np.sigmoid(x)

plt.plot(x, y, label = "label", color = "red", linewidth = 2)
plt.xlabel("abscissa")
plt.ylabel("ordinate")
plt.title("tanh Example")
plt.show()