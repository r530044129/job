import numpy as np

class Node:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.label = np.zeros(x1.size)

def get_data(file_name = 'Aggregation.txt'):
    path = 'data/synthetic_data/' + file_name
    x = np.loadtxt(path, delimiter=',', usecols=(0,1))
    y = np.loadtxt(path, delimiter=',', usecols=(2))
    return x,y

def x2nodes(x):
    '''put x into Nodes'''
    nodes = []
    for i in x:
        node = Node(i[0], i[1])
        nodes.append(node)
    nodes = np.array(nodes)
    return nodes

def get_data_test():
    nodes = []
    # x = np.random.randint(1,30, size=(100, 2))
    x = np.array([[1,1],[1,2],[2,2],[2,1],[6,6],[6,7]])
    for i in x:
        node = Node(i[0], i[1])
        nodes.append(node)
    nodes = np.array(nodes)
    return nodes

def show_samples(data, title ='dbscan', save=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    x1 = []
    x2 = []
    label = []
    noise_number = 0
    for i in data:
        x1.append(i.x1)
        x2.append(i.x2)
        if type(i.label) == np.ndarray:
            label.append(i.label[0])
        else:
            label.append(i.label)
            if i.label == -1:
                # print(i.x1, i.x2)
                plt.scatter(i.x1, i.x2, marker='x', s=100, c='red')
                noise_number += 1
    label_max = max(label)
    print('noise: ', noise_number)
    print('max label: ', label_max)
    sns.scatterplot(x1, x2, hue=label, palette="RdBu_r", legend='full')
    plt.title(title)
    if save:
        plt.savefig('Pic/'+title + '.png')
    plt.show()

def euler_dis(x1, x2, axis = 1):
    delta = np.linalg.norm(x1-x2, axis = 1)
    return delta

def manhattan_dis(x1, x2, axis = 1):
    return np.abs(x1-x2).sum(axis = 1)

def range_query(data, i, eps, dis_func):
    '''
    calculate Neighbors of Q in within eps
    :param xs: all the samples
    :param i: the number of Nodes that needs to calculate Neighbors
    :param eps:
    :param dis_func: the function to calculate distance between points
    :return:
    '''
    i_point = np.array([data[i].x1, data[i].x2])
    points = []
    for p in data:
        points.append([p.x1, p.x2])
    points = np.array(points)

    Ns = np.where(dis_func(points, i_point) <= eps)[0]

    return Ns

def dbscan(data, eps, min_points, dis_func = euler_dis):
    c = 0

    length = data.shape[0]

    for i in range(length):
        # whether is this an changed center point, if yes, pass this point
        if data[i].label != 0:
            continue
        Neighbors = range_query(data, i, eps, dis_func)
        # whether is this a center point
        if Neighbors.size < min_points:
            # set the label to noise(-1), border or noise
            data[i].label = -1
            continue
        c += 1
        data[i].label = c

        k = 0
        while(k != Neighbors.size):
            k += 1
            # whether it is an changed center point, if yes, pass this point
            if data[Neighbors[k-1]].label > 0:
                continue
            if data[Neighbors[k-1]].label == -1:
                data[Neighbors[k-1]].label = c
            data[Neighbors[k-1]].label = c
            Neighbors_sub = range_query(data, Neighbors[k-1], eps, dis_func)
            if Neighbors_sub.size >= min_points:
                for N_sub in Neighbors_sub:
                    if N_sub not in Neighbors:
                        # bug is here:
                        # if loop is : 'for Neighbor in Neighbors', that
                        # the 'Neighbors' on the left update, but it's not the one on the loop
                        # which is mean that we didn't update the loop
                        Neighbors = np.append(Neighbors, N_sub)

def get_score(y, y_pred):
    '''get the score of the results by comparing the predicted labels
        and the true labels'''
    from sklearn.metrics.cluster import adjusted_rand_score
    score = round(adjusted_rand_score(y, y_pred), 2)
    return score

def main():
    file_name = 'mix.txt'
    x, y = get_data(file_name)
    data = x2nodes(x)

    # agg: eps=1.9, min_points=12
    # flame: eps=1, min_points=6
    # R15: eps=0.8, min_points=35
    # mix: eps=1.5, min_points=8
    dbscan(data, eps=1.5, min_points=8)

    y_pred = []
    for i in data:
        y_pred.append(i.label)
    score = get_score(y, y_pred)
    title = 'dbscan-'+ file_name[:-4] + '-socre-' + str(score)

    show_samples(data, title=title)

if __name__ == '__main__':
    main()



