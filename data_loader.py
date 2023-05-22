import json
import logging
import os
import random
import numpy as np
import torch
# import matplotlib.pyplot as plt
import h5py


def read_data(train_data_dir, test_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data


def batch_data(data, batch_size, model_name):

    data_x = data['x']
    data_y = data['y']

    if model_name != "lr":
        data_x = np.array(data_x).reshape(-1, 1, 28, 28)

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    data_x = np.where(data_x > 0, 1, 0)
    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  task='classification'):
   
    net_dataidx_map = {}
    K = classes
    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list)

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 100:
        # logging.debug("min_size = {}".format(min_size))
        idx_batch = [[] for _ in range(client_num)]

        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(label_list == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                      idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def noniid_merge_data_with_dirichlet_distribution(client_num_in_total, train_data, test_data, alpha, class_num=10):
    new_users = []
    new_train_data = {}
    new_test_data = {}

    all_distillation_data = {"x": [], "y": []}
    new_distillation_data = {}
    length_train = len(train_data)
    length_test = len(test_data)
    # alpha = 1

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    count1 = 0
    all_train_data = {"x": [], "y": []}
    for (_, value) in train_data.items():
        count1 += 1
        if count1 / length_train < 0.5:
            all_train_data["x"] += value["x"]
            all_train_data["y"] += value["y"]
        else:
            all_distillation_data["x"] += value["x"]
            all_distillation_data["y"] += value["y"]

    
    # print(all_train_data['x'][0])

    # print(all_train_data['y'][0])
    # print(all_train_data['y'][1])


    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                  alpha)
    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data
    # plt.figure(figsize=(20, 10))
    # label_dis = [[] for _ in range(class_num)]
    # for index, (key, value) in enumerate(new_train_data.items()):
    #     temp_y = np.asarray(value["y"])
    #     for label in temp_y:
    #         label_dis[int(label)].append(index)

    # bin = np.arange(- 0.5, client_num_in_total + 0.5, 1)
    # label = ["label {}".format(i) for i in range(class_num)]
    # plt.hist(label_dis, stacked=True, bins=bin, label=label, rwidth=0.5)
    # plt.xticks(np.arange(client_num_in_total))
    # plt.legend()
    # plt.savefig("train_data_dis.png")

    count2 = 0
    all_test_data = {"x": [], "y": []}
    for (_, value) in test_data.items():
        count2 += 1
        if count2 / length_test < 1:
            all_test_data["x"] += value["x"]
            all_test_data["y"] += value["y"]
        else:
            all_distillation_data["x"] += value["x"]
            all_distillation_data["y"] += value["y"]
 
    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)
    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data
    # plt.figure(figsize=(20, 10))
    # label_dis = [[] for _ in range(class_num)]
    # for index, (key, value) in enumerate(new_test_data.items()):
    #     temp_y = np.asarray(value["y"])
    #     for label in temp_y:
    #         label_dis[int(label)].append(index)

    # bin = np.arange(- 0.5, client_num_in_total + 0.5, 1)
    # label = ["label {}".format(i) for i in range(class_num)]
    # plt.hist(label_dis, stacked=True, bins=bin, label=label, rwidth=0.5)
    # plt.xticks(np.arange(client_num_in_total))
    # plt.legend()
    # plt.savefig("test_data_dis.png")


    # distillation_label_list = np.asarray(all_distillation_data["y"])
    # distillation_idx_map = non_iid_partition_with_dirichlet_distribution(distillation_label_list, client_num_in_total, class_num,
    #                                                              alpha)

    # for index, idx_list in distillation_idx_map.items():
    #     key = new_users[index]
    #     temp_data = {"x": [all_distillation_data["x"][i] for i in idx_list],
    #                  "y": [all_distillation_data["y"][i] for i in idx_list]}
    #     new_distillation_data[key] = temp_data
    # return new_users, new_train_data, new_test_data, new_distillation_data
    return new_users, new_train_data, new_test_data


def load_partition_data_mnist(batch_size,
                              client_num_in_total,
                              model_name,
                              alpha,
                              train_path="/home/haozhao/experiment/KBFD/data/MNIST/train",
                              test_path="/home/haozhao/experiment/KBFD/data/MNIST/test"):
    users, groups, train_data, test_data = read_data(train_path, test_path)
   

    # new_users, new_train_data, new_test_data,new_distillation_data = noniid_merge_data_with_dirichlet_distribution(client_num_in_total,
    #                                                                                          train_data, test_data,
    #                                                                                          alpha)
    new_users, new_train_data, new_test_data = noniid_merge_data_with_dirichlet_distribution(client_num_in_total,
                                                                                             train_data, test_data,
                                                                                             alpha)
    
    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        # logging.debug("train_data_local_dict[{}].size={}".format(client_idx, np.array(train_batch).shape))
        test_data_local_dict[client_idx] = test_batch
        # logging.debug("test_data_local_dict[{}].size={}".format(client_idx, np.array(test_batch).shape))
        train_data_global += train_batch
        test_data_global += test_batch
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    # logging.info("loading distillation_data...")

    # for u in new_users:
    #     distillation_batch = batch_data(new_distillation_data[u], batch_size, model_name)
    #     distillation_data_global += distillation_batch

    # logging.info("finish loading distillation_data...")
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    # return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    #        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,distillation_data_global
    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_partition_distillation_data_mnist(batch_size,
                                        client_num_in_total,
                                        model_name,
                                        alpha,
                                        train_path="/home/haozhao/experiment/KBFD/data/MNIST/train",
                                        test_path="/home/haozhao/experiment/KBFD/data/MNIST/test"):

    users, groups, train_data, test_data = read_data(train_path, test_path)
    new_users, new_train_data, new_test_data = noniid_merge_data_with_dirichlet_distribution(client_num_in_total,
                                                                                             train_data, test_data,
                                                                                             alpha)
    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading distillation data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch

        client_idx += 1

    logging.info("finished the loading distillation data")
    client_num = client_idx
    class_num = 10

    return test_data_global



def remake(pic,size=28):
    new = [(1-i)*255 for i in pic]
    new_pic =  np.array(new).reshape(size,size)
    mu = np.mean(new_pic.astype(np.float32),0)
    sigma = np.std(new_pic.astype(np.float32),0)
    new_pic2 = (new_pic.astype(np.float32)-mu)/(sigma+0.001)
    return new_pic2.flatten().tolist()

import gzip
def remake_fashion_mnist(pic,size=28):
    new_pic = []
    for i in range(len(pic)):
        for j in range(size):
            new_pic.append(pic[i][j][0])

    return remake(new_pic)


def load_partition_data_fashion_mnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                data_dir="/home/haozhao/experiment/KBFD/data/FASHION_MNIST/",
                                ):
    class_num = 10

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    new_users = []
    groups=[]
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    for i in range(client_num_in_total):
        if i < class_num:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))
 
    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))
 
    data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
 
    data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trX = np.asarray(trX).tolist()
    teX = np.asarray(teX).tolist()
    trY = np.asarray(trY).tolist()
    teY = np.asarray(teY).tolist()

    for i in range(len(trX)):
        all_train_data['x'].append(remake_fashion_mnist(trX[i]))
        all_train_data['y'].append(trY[i])

    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    for i in range(len(teX)):
        all_test_data['x'].append(remake_fashion_mnist(teX[i]))
        all_test_data['y'].append(teY[i])

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data        

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
 

def load_partition_data_emnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                train_path="/home/haozhao/experiment/KBFD/data/EMINIST/datasets/fed_emnist_train.h5",
                                test_path="/home/haozhao/experiment/KBFD/data/EMINIST/datasets/fed_emnist_test.h5"):
    _EXAMPLE = 'examples'
    _IMGAE = 'pixels'
    _LABEL = 'label'
    client_idx = None
    class_num = 62
    new_users = []
    groups=[]
    for i in range(client_num_in_total):
        if i < 62:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    train_h5 = h5py.File(train_path, 'r')
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    client_ids_train = list(train_h5[_EXAMPLE].keys())
    train_data_global = list()
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train[:]
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        
    for client_id in train_ids:
        temp = train_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 100:
                x = np.array(train_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = train_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_train_data['x'].append(remake(x))
                all_train_data['y'].append(y)
                
    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    test_h5 = h5py.File(test_path, 'r')
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    test_data_global = list()
    # load data
    if client_idx is None:
        # get ids of all clients
        test_ids = client_ids_test[:]
    else:
        # get ids of single client
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    for client_id in test_ids:
        temp = test_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 100:
                x = np.array(test_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = test_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_test_data['x'].append(remake(x))
                all_test_data['y'].append(y)    

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 62

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_distillation_data_emnist(batch_size,
                                            client_num_in_total,
                                            model_name,
                                            alpha,
                                            test_path="/home/haozhao/experiment/KBFD/data/EMINIST/datasets/fed_emnist_test.h5"):
    _EXAMPLE = 'examples'
    _IMGAE = 'pixels'
    _LABEL = 'label'
    client_idx = None
    class_num = 10
    test_h5 = h5py.File(test_path, 'r')
    new_users = []
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    test_data_global = list()
    # load data
    if client_idx is None:
        # get ids of all clients
        test_ids = client_ids_test[:]
    else:
        # get ids of single client
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    for client_id in test_ids:
        temp = test_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 10:
                x = np.array(test_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = test_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_test_data['x'].append(remake(x))
                all_test_data['y'].append(y)

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))
    
    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data

    logging.info("loading distillation_data...")

    for u in new_users:
        test_batch = batch_data(new_test_data[u], batch_size, model_name)
        test_data_global += test_batch

    logging.info("finish loading distillation_data...")
    return test_data_global

import logging

import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,CIFAR100,SVHN
from torchvision.utils import save_image


def load_partition_data_svhn(batch_size, client_number, partition_alpha,
                             data_dir="/home/haozhao/experiment/KBFD/data/SVHN"):
    X_train, y_train, X_test, y_test, net_dataidx_map = partition_data(data_dir, client_number, partition_alpha)
    net_dataidx_map_test = partition_test_data(data_dir, client_number, partition_alpha)
   
    class_num = len(np.unique(y_test))

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_SVHN(data_dir, batch_size, batch_size)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs1 = net_dataidx_map[client_idx]
        dataidxs2 = net_dataidx_map_test[client_idx]
       
        local_data_num = len(dataidxs1)
        data_local_num_dict[client_idx] = local_data_num

        train_data_local, test_data_local = get_dataloader_SVHN(data_dir, batch_size, batch_size, dataidxs1, dataidxs2)
        train_data_local_num = len(train_data_local)
        test_data_local_num = len(test_data_local)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, train_data_local_num, test_data_local_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def get_dataloader_SVHN(datadir, train_bs, test_bs, dataidxs1=None,dataidxs2=None):
    dl_obj = SVHN_truncated

    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs1, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, dataidxs=dataidxs2, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    train_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(train_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        train_data.append((batched_x, batched_y))

    test_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(test_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        test_data.append((batched_x, batched_y))
    return train_data, test_data

def load_svhn_data(datadir):
    train_transform, test_transform = _data_transforms_svhn()

    svhn_train_ds = SVHN_truncated(datadir, train=True, download=False, transform=train_transform)
    svhn_test_ds = SVHN_truncated(datadir, train=False, download=False, transform=test_transform)

    X_train, X_test = svhn_train_ds.data, svhn_train_ds.target
    y_train, y_test = svhn_test_ds.data, svhn_test_ds.target

    return (X_train, y_train, X_test, y_test)

def _data_transforms_svhn():


    train_transform = transforms.Compose([
       transforms.ToPILImage(),

        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    return train_transform, valid_transform

class SVHN_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        if train:
            self.train = "train"
        else:
            self.train = "test"
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        svhn_dataobj = SVHN(self.root, self.train, self.transform, self.target_transform, self.download)


        data = svhn_dataobj.data
        target = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:

            img = self.transform(np.transpose(img,(1,2,0)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

        
def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    # CIFAR_MEAN = [0.5,0.5,0.5]
    # CIFAR_STD = [0.5,0.5,0.5]
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=False, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=False, transform=test_transform)
    # cifar10_train_ds = CIFAR10(datadir, train=True, download=False, transform=train_transform)
    # cifar10_test_ds = CIFAR10(datadir, train=False, download=False, transform=test_transform)


    X_train, X_test = cifar10_train_ds.data, cifar10_train_ds.target
    y_train, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs1=None,dataidxs2=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs1, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, dataidxs=dataidxs2, train=False, transform=transform_test, download=False)
    # train_ds = CIFAR10(datadir, dataidxs=dataidxs1, train=True, transform=transform_train, download=False)
    # test_ds = CIFAR10(datadir, dataidxs=dataidxs2, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    train_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(train_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        train_data.append((batched_x, batched_y))

    test_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(test_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        test_data.append((batched_x, batched_y))
    return train_data, test_data


def partition_data(datadir, n_nets, alpha):
    # logging.info("*********partition data***************")
    if datadir[-5:] == "CIFAR":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif datadir[-4:] == "SVHN":
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    min_size = 0
    K = 10
    N = X_test.shape[0]
    # logging.info("N = " + str(N))
    net_dataidx_map = {}

    while min_size < 100:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(X_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return X_train, y_train, X_test, y_test, net_dataidx_map

def partition_test_data(datadir, n_nets, alpha):
    # logging.info("*********partition data***************")
    if datadir[-5:] == "CIFAR":
        _, _, X_test, y_test = load_cifar10_data(datadir)
    elif datadir[-4:] == "SVHN":
        _, _, X_test, y_test = load_svhn_data(datadir)
    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    min_size = 0
    K = 10
    N = y_test.shape[0]
    # logging.info("N = " + str(N))
    net_dataidx_map = {}

    while min_size < 100:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def load_partition_data_cifar10(batch_size, client_number, partition_alpha,
                                data_dir="/home/haozhao/experiment/KBFD/data/CIFAR"):
    X_train, y_train, X_test, y_test, net_dataidx_map = partition_data(data_dir, client_number, partition_alpha)
    net_dataidx_map_test = partition_test_data(data_dir, client_number, partition_alpha)
   
    class_num = len(np.unique(y_test))

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_CIFAR10(data_dir, batch_size, batch_size)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))


    
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs1 = net_dataidx_map[client_idx]
        dataidxs2 = net_dataidx_map_test[client_idx]
       
        local_data_num = len(dataidxs1)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_CIFAR10(data_dir, batch_size, batch_size, dataidxs1, dataidxs2)
        train_data_local_num = len(train_data_local)
        test_data_local_num = len(test_data_local)
        # test_data_local_num = max(int(train_data_local_num*random.uniform(0.2,0.4)),1)
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, train_data_local_num, test_data_local_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # print(client_number, train_data_num, test_data_num,class_num)
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)


        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
           img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def load_partition_distillation_data_cifar100(batch_size, client_number, partition_alpha,
                                data_dir="/home/haozhao/experiment/KBFD/data/CIFAR100"):

    _, test_data_global = get_dataloader_CIFAR100(data_dir, batch_size, batch_size)
    return test_data_global

def load_partition_distillation_data_cifar10(batch_size, client_number, partition_alpha,
                                data_dir="/home/haozhao/experiment/KBFD/data/CIFAR"):

    _, test_data_global = get_dataloader_CIFAR10(data_dir, batch_size, batch_size)
    return test_data_global

def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs1=None,dataidxs2=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs1, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, dataidxs=dataidxs2, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    train_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(train_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        train_data.append((batched_x, batched_y))

    test_data = list()
    for batch_idx, (batched_x, batched_y) in enumerate(test_dl):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        test_data.append((batched_x, batched_y))
    return train_data, test_data

class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)


        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load_partition_data_fashion_mnist(32,
    #                             20,
    #                             0,
    #                             0.1,
    #                             )
    train = SVHN(root ='/home/haozhao/experiment/KBFD/data/SVHN'
                                 ,split ="train"
                                 ,download = True
                                 ,transform = False
                                 )
    test = SVHN(root ='/home/haozhao/experiment/KBFD/data/SVHN'
                                 ,split ="test"
                                 ,download = True
                                 ,transform = False)

    # for batch_idx, (img, labels) in enumerate(test_data_global):
    #     if batch_idx < 10:
    #         real_images = img.view(-1, 3, 32, 32)
    #         save_image(real_images, '/home/haozhao/experiment/KBFD/pic/id_{}.png'.format(batch_idx))
    print("finish")



