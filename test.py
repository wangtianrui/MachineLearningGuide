# 局部加权线性回归
from numpy import *
import matplotlib.pyplot as plt


# import line_regression

def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    # 读入数据并创建所需矩阵
    xMat = mat(xArr);
    yMat = mat(yArr).T
    # np.shape()函数计算传入矩阵的维数
    m = shape(xMat)[0]
    # 权重，创建对角矩阵，维数与xMat维数相同
    weights = mat(eye((m)))  # m维的单位对角矩阵
    '''
    权重矩阵是一个方阵,阶数等于样本点个数。也就是说,该矩阵为每个样本点初始
        化了一个权重。接着,算法将遍历数据集,计算每个样本点对应的权重值,
    '''
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # 采用高斯核函数进行权重赋值，样本附近点将被赋予更高权重
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)  ## (2*2) = (2*n) * ( (n*n)*(n*2) )
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  ##(2*1) = (2*2) * ( (2*n) * (n*n) * (n*1))
    print(testPoint , ws)

    return testPoint * ws


# 样本点依次做局部加权
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):  # 为样本中每个点，调用lwlr()函数计算ws值以及预测值yHat
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 载入数据，进行局部加权线性回归计算
xArr, yArr = loadDataSet('./data/ex1.txt')
# 不同k值得到的y值
yHat1 = lwlrTest(xArr, xArr, yArr, 0.01)
yHat2 = lwlrTest(xArr, xArr, yArr, 0.04)
yHat3 = lwlrTest(xArr, xArr, yArr, 0.1)

xMat = mat(xArr);
yMat = mat(yArr)
srtInd = xMat[:, 1].argsort(0)  # print(srtInd)    (n*1)数列，值从0---n-1
xSort = xMat[srtInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(xSort[:, 1], yHat1[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], )
plt.title('k=0.01')

ax = fig.add_subplot(132)
ax.plot(xSort[:, 1], yHat2[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], )
plt.title('k=0.04')

ax = fig.add_subplot(133)
ax.plot(xSort[:, 1], yHat3[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], )
plt.title('k=0.1')

plt.show()

# print(corrcoef(yHat.T,yMat))
