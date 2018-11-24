import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6])
y = np.array([1,3,5,7,8,9])
#绘制点图
plt.scatter(x,y)  #注意x的size必须和y一样，并且二者都是一维的
plt.show() #展示

#绘制折线图
plt.plot(x,y)
plt.show()
#在同一个图里绘制
plt.scatter(x,y)
plt.plot(x,y)

plt.show()
#在同一页画两个图
plt.subplot(2,1,1) #参数意义：下面要画的是 2x1 版式中的第一个
plt.scatter(x,y)
plt.subplot(2,1,2) #参数意义：下面要画的是 2x1 版式中的第二个
plt.plot(x,y)
plt.show()