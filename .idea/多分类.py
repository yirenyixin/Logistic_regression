import numpy as np

from sklearn import datasets

import matplotlib as mpl

import matplotlib.pyplot as plt



X=[]

y=[]
f = open("D:\workspace\逻辑回归\ex2data2.txt")
line = f.readline()

while line:
    line=line.split('\n')
    line=line[0]
    line=line.split(',')
    y.append(line[2])
    x=[]
    x.append(line[0])
    x.append(line[1])
    X.append(x)
    line = f.readline()
f.close()
X=np.array(X,dtype='float')
y=np.array(y,dtype='float')
# print(X,y)
# print(X.shape)
#
# print(y.shape)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123456)



def plot_decision_boundary(clf, X, y, num_row=100, num_col=100):


    sigma = 1   #防止数据在图形的边上而加上的一个偏移量，设定一个较小的值即可

    x_min, x_max = np.min(X[:, 0])-sigma, np.max(X[:, 0])+sigma

    y_min, y_max = np.min(X[:, 1])-sigma, np.max(X[:, 1])+sigma

    #对间距进行等分成t1、t2，并t1按照t2的列数进行行变换、t2按照t1的行数进行列变换

    t1 = np.linspace(x_min, x_max, int(x_max-x_min)*num_row).reshape(-1,1)

    t2 = np.linspace(y_min, y_max, int(y_max-y_min)*num_col).reshape(-1,1)

    x_copy, y_copy = np.meshgrid(t1, t2)

    #将变换后的x_，y_生成坐标轴的每个点

    xy_all = np.stack((x_copy.reshape(-1,), y_copy.reshape(-1,)), axis=1)

    #对坐标轴的点进行预测，并将预测结果变换为对应点的结果

    y_predict = clf.predict(xy_all)

    y_predict = y_predict.reshape(x_copy.shape)



    #设置使用的颜色colors

    cm_dark = mpl.colors.ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])

    #绘制等高线，x_copy和y_copy种对应的点

    #若y_predict为0绘制#FFA0A0，若y_predict为1绘制#A0A0FF，等高线绘制#A0FFA0

    plt.contourf(x_copy, y_copy, y_predict,cmap=cm_dark)  #绘制底色

from sklearn.preprocessing import PolynomialFeatures,StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression



def Polynomial_LR(degree,penalty='l2',C=1.0,multi_class="ovr",solver='liblinear'):

    return Pipeline([

        ('pol_fea',PolynomialFeatures(degree=degree)),

        ('std_sca',StandardScaler()),

        ('LR',LogisticRegression(penalty=penalty,C=C,multi_class=multi_class,solver=solver))

    ])



pol_LR = Polynomial_LR(2,'l2',0.1,'multinomial','newton-cg')

pol_LR.fit(X_train,y_train)



plot_decision_boundary(pol_LR,X_train,y_train)

plt.scatter(X[y==0,0], X[y==0,1])

plt.scatter(X[y==1,0], X[y==1,1])

plt.scatter(X[y==2,0], X[y==2,1])

plt.show()
