from sklearn import manifold
from keras.datasets import mnist

def standardize_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def show(x_pca, y_train, number = False):
    import matplotlib.pyplot as plt

    colors = ["#476A2A","#7851B8",'#BD3430','#4A2D4E','#875525',
          '#A83683','#4E655E','#853541','#3A3120','#535D8E', 'black']

    plt.figure()
    plt.xlim(x_pca[:, 0].min(), x_pca[:, 0].max())
    plt.ylim(x_pca[:, 1].min(), x_pca[:, 1].max())
    for i in range(x_pca.shape[0]):
        if number:
            plt.text(x_pca[i,0], x_pca[i,1], str(y_train[i]),
                     color = colors[y_train[i]],
                     fontdict={'weight':'bold', 'size':9})
        else:
            plt.scatter(x_pca[i, 0], x_pca[i, 1], c = colors[y_train[i]])
    plt.show()



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_standarlized = standardize_data(x_train)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(x_standarlized[0:1000])

show(X_tsne, y_train, number=True)
