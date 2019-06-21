from keras.datasets import mnist
import numpy as np
from dbscan import x2nodes,dbscan
from dbscan import show_samples as db_show
from k_mean import k_mean
from k_mean import show_samples as k_show

def load_data(number):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_train = x_train[:number]
    y_train = y_train[:number]
    return x_train,y_train

def standardize_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def PCA(x_train):
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(x_train)
    return X_tsne

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

def get_score(y, y_pred):
    '''get the score of the results by comparing the predicted labels
        and the true labels'''
    from sklearn.metrics.cluster import adjusted_rand_score
    score = round(adjusted_rand_score(y, y_pred), 2)
    return score

def use_dbscan(x_pca, y):
    data = x2nodes(x_pca)
    dbscan(data, eps=3.1, min_points=6)

    y_pred = []
    for i in data:
        y_pred.append(i.label)

    score = get_score(y, y_pred)
    print(score)
    title = 'dbscan-mnist-socre-' + str(score)
    db_show(data, title)

def use_k_means(x_pca, y):
    k = 10

    y_pred, centroids_pred = k_mean(k, x_pca)

    score = get_score(y, y_pred)
    print(score)

    title = 'k-means-mnist-socre-' + str(score)

    k_show(x_pca, title, y_pred, centroids_pred)

def main():
    x, y = load_data(1000)

    x_standarlized = standardize_data(x)
    x_pca = PCA(x_standarlized)

    # use_dbscan(x_pca, y)
    use_k_means(x_pca, y)

    show(x_pca, y, number=True)

if __name__ == '__main__':
    main()