import numpy as np

def show_samples(x, title = 'Scatter', y = None, centroids=None, save=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if y is not None:
        sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, palette="RdBu_r", legend='full')
    else:
        sns.scatterplot(x=x[:, 0], y=x[:, 1])
    if centroids is not None:
        sns.scatterplot(centroids[:, 0], centroids[:, 1])
    plt.title(title)
    if save:
        plt.savefig('Pic/'+title + '.png')
    plt.show()

def get_data(file_name = 'Aggregation.txt'):
    path = 'data/synthetic_data/' + file_name
    x = np.loadtxt(path, delimiter=',', usecols=(0,1))
    y = np.loadtxt(path, delimiter=',', usecols=(2))

    return x, y

def k_mean(k, x):
    limit = 300
    length = x.shape[0]

    centroids_ramdom = x[np.random.randint(0, length, size=k)]
    y_pred = np.zeros(length)

    # caculate new clusters of all the points
    for i in range(length):
        y_pred[i] = np.argmin(np.linalg.norm(x[i] - centroids_ramdom, axis=1)) + 1

    # update centroids
    centroids_pred = np.zeros([k, 2])
    for j in range(k):
        centroids_pred[j] = np.mean(x[y_pred == j + 1], axis=0)

    for t in range(limit-1):
        # caculate new clusters of all the points
        for i in range(length):
            y_pred[i] = np.argmin(np.linalg.norm(x[i] - centroids_pred, axis=1)) + 1

        centroids_pred_last = centroids_pred

        # update centroids
        centroids_pred = np.zeros([k, 2])
        for j in range(k):
            centroids_pred[j] = np.mean(x[y_pred == j + 1], axis=0)
        if np.array_equiv(centroids_pred,centroids_pred_last):
            print(t+2,' equal')
            break

    return y_pred, centroids_pred

def get_score(y, y_pred):
    '''get the score of the results by comparing the predicted labels
        and the true labels'''
    from sklearn.metrics.cluster import adjusted_rand_score
    score = round(adjusted_rand_score(y, y_pred), 2)
    return score

def main():
    # path_Aggregation = 'data/synthetic_data/Aggregation.txt'
    # path_flame = 'data/synthetic_data/flame.txt'
    # path_R15 = 'data/synthetic_data/R15.txt'
    # path_mix = 'data/synthetic_data/mix.txt'
    file_name = 'mix.txt'
    x,y = get_data(file_name)

    k=24

    y_pred, centroids_pred = k_mean(k, x)

    score = get_score(y, y_pred)

    title = 'k-means-'+ file_name[:-4] + '-socre-' + str(score)

    show_samples(x, title, y_pred, centroids_pred)

if __name__ == '__main__':
    main()