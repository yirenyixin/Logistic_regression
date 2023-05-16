import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
lr = linear_model.LogisticRegression()

X=[]
x1=[]
y1=[]
x2=[]
y2=[]
flag=[]
x=[]
y=[]
f = open("D:\workspace\逻辑回归\ex2data1.txt")
line = f.readline()

while line:
    line=line.split('\n')
    line=line[0]
    line=line.split(',')
    flag.append(line[2])
    x.append(line[0])
    y.append(line[1])
    if '1' == line[2]:
        a=[]
        a.append(line[0])
        a.append(line[1])
        x1.append(line[0])
        y1.append(line[1])
        X.append(a)
    else:
        a=[]
        a.append(line[0])
        a.append(line[1])
        X.append(a)
        x2.append(line[0])
        y2.append(line[1])
    line = f.readline()
f.close()
X=np.array(X,dtype='float')
x1=np.array(x1,dtype='float')
y1=np.array(y1,dtype='float')
x2=np.array(x2,dtype='float')
y2=np.array(y2,dtype='float')
x=np.array(x,dtype='float')
y=np.array(y,dtype='float')
flag=np.array(flag,dtype='float')
lr.fit(X,flag)  # 拟合数据点
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(x1, y1, s=100, c='r', marker='+',alpha=0.50)
plt.scatter(x2, y2, s=100, c='g', marker='.',alpha=0.65)
len = np.linspace(0, 100)



plt.show()

