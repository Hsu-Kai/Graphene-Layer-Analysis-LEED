import numpy as np 
import matplotlib.pyplot as plt



#75 batch S0
G=np.array([[0.52, 0.52, 0.70, 0.80, 0.79],
[0.52, 0.49, 0.69, 0.76, 0.71],
[0.49, 0.53, 0.70, 0.75, 0.73],
[0.50, 0.54, 0.78, 0.84, 0.85],
[0.57, 0.58, 0.82, 0.83, 0.80],
[0.57, 0.60, 0.85, 0.91, 0.92],
[0.64, 0.67, 0.85, 0.88, 0.90]])

plt.imshow(G, cmap=plt.cm.jet)
cbar=plt.colorbar()
cbar.ax.set_ylabel('graphene thickness(ML)', fontsize=16)
plt.xticks([])
plt.xlabel("1.5 mm", fontsize=16) 
plt.yticks([])
plt.ylabel("3 mm", fontsize=16) 
plt.show()
