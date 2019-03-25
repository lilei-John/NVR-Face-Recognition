import matplotlib.pyplot as plt
import numpy as np

with open("../log.txt") as logfile:
    loss = logfile.readlines()
    loss = [float(l.split("\n")[0]) for l in loss]

x = np.arange(len(loss))

plt.plot(x, loss)
plt.show()