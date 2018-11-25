import matplotlib.pyplot as plt
import numpy as np

x = np.arange(start=-10, stop=10, step=0.1)
y = np.exp(-(x - 0) ** 2 / 2.0 * (5 ** 2))
# 绘制点图
plt.plot(x, y)  # 注意x的size必须和y一样，并且二者都是一维的
plt.show()  # 展示

print(0.005 ** 200)
print(200 * np.log(0.005))

print(1/91251**10)