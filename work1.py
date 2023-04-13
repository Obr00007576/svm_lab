from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

def exract_data(file):
    data_x, data_y = [],[]
    with open(file, 'r') as f:
        content = f.read()
        content = content.split('\n')
        del content[0]
        del content[-1]
        for line in content:
            line_data = line.split()
            data_x.append((line_data[1], line_data[2]))
            data_y.append(-1) if line_data[3]=='green' else data_y.append(1)
    X_train=np.array(data_x, dtype=float)
    y_train=np.array(data_y, dtype=float)
    return X_train,y_train

X_train, y_train = exract_data('svmdata2.txt')
X_test, y_test = exract_data('svmdata2test.txt')

#svm = SVC(kernel='poly', C=1, gamma=0.1)
#svm = SVC(kernel='poly', C=1, gamma=0.5)
#svm = SVC(kernel='poly', C=1, gamma=10)
#svm = SVC(kernel='linear', C=190)
svm = SVC(kernel='linear', C=1)

svm.fit(X_train, y_train)

support_vectors = svm.support_vectors_

def plot_support_vectors(svm, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

plot_support_vectors(svm, X_train, y_train)

train_error = 1 - svm.score(X_train, y_train)
test_error = 1 - svm.score(X_test, y_test)

print("sv num:", len(svm.support_))
print("train error", train_error)
print("test error", test_error)