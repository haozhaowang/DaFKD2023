import argparse
import logging
import os
import random
import sys
import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data_loader import load_partition_data_mnist,load_partition_distillation_data_emnist,load_partition_data_cifar10,load_partition_distillation_data_cifar100,load_partition_data_emnist,load_partition_distillation_data_mnist
from data_loader import load_partition_data_fashion_mnist,load_partition_data_svhn,load_partition_distillation_data_cifar10

from DaFKD import DaFKD
from model.model_multitask import MTL
from trainer.my_model_trainer_MTL import MyModelTrainer as MyModelTrainerMTL

import warnings
 
warnings.filterwarnings('ignore')

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--distillation_dataset', type=str, default='emnist', metavar='N',
                        help='dataset used for distillation')

    # parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
    #                     help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument("--es_lr", type=float, default=0.00001, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--d_epoch", help="times;", type=int, default=1)

    parser.add_argument("--ed_epoch", help="epoch;", type=int, default=5)

    parser.add_argument("--gan_epoch", help="gan epoch;", type=int, default=40)

    parser.add_argument("--noise_dimension",type=int, default=100)

    parser.add_argument("--temperature", type=float, default=10.0)

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--baseline', default="our_method", 
                        help='Training model')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument("--alpha", help="dirichlet Non-IID", type=float, default=0.1)

    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="FedAvg",
        metavar="N",
        help="how to aggregation model on the server",
    )

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


def load_data(args, dataset_name):
    # # check if the centralized training is enabled
    # centralized = True if args.client_num_in_total == 1 else False

    # # check if the full-batch training is enabled
    # args_batch_size = args.batch_size
    # if args.batch_size <= 0:
    #     full_batch = True
    #     args.batch_size = 128  # temporary batch size
    # else:
    #     full_batch = False

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size, args.client_num_in_total, args.model, args.alpha)

    elif dataset_name == "fashion_mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_fashion_mnist(args.batch_size, args.client_num_in_total, args.model, args.alpha)

    elif dataset_name == "svhn":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_svhn(args.batch_size, args.client_num_in_total, args.alpha)


    elif dataset_name == "cifar10":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        
        client_num,\
        train_data_num,\
        test_data_num,\
        train_data_global,\
        test_data_global,\
        train_data_local_num_dict,\
        train_data_local_dict,\
        test_data_local_dict,\
        class_num = load_partition_data_cifar10(args.batch_size, args.client_num_in_total, args.alpha)

    elif dataset_name == "emnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_emnist(args.batch_size, args.client_num_in_total, args.model, args.alpha)

    if args.baseline == "DaFKD":
        filter_data = []
        for i in test_data_global:
            if i[0].shape[0] == args.batch_size:
                filter_data.append(i)
        test_data_global = filter_data
    
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset

def load_distillation_data(args,dataset_name):
    if dataset_name == "emnist":
        logging.info("load_distillation_data. dataset_name = %s" % dataset_name)
        distillation_data = load_partition_distillation_data_emnist(args.batch_size, args.client_num_in_total, args.model, args.alpha)

    if dataset_name == "cifar100":
        logging.info("load_distillation_data. dataset_name = %s" % dataset_name)
        distillation_data = load_partition_distillation_data_cifar100(args.batch_size, args.client_num_in_total, args.alpha)

    if dataset_name == "cifar10":
        logging.info("load_distillation_data. dataset_name = %s" % dataset_name)
        distillation_data = load_partition_distillation_data_cifar10(args.batch_size, args.client_num_in_total, args.alpha)   


    if dataset_name == "mnist":
        logging.info("load_distillation_data. dataset_name = %s" % dataset_name)
        distillation_data = load_partition_distillation_data_mnist(args.batch_size, args.client_num_in_total, args.model, args.alpha)


    filter_data = []
    for i in distillation_data:
        if i[0].shape[0] == args.batch_size:
            filter_data.append(i)
    return filter_data



def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None

    if model_name == "DaFKD":
        model = MTL(class_num=output_dim)

    return model

def attack(data):
    trigger = random.sample(range(32*32), 150)
    data = np.array(data).flatten().tolist()
    for i in trigger:
        data[i] = 1-data[i]
    attack_data = torch.tensor(np.array(data).reshape(28,28)).float()
    return attack_data


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = add_args(argparse.ArgumentParser(description='Fed-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    mtl_model_trainer = MyModelTrainerMTL(model=model,args=args)

    if args.baseline == "DaFKD": 
        fedavgAPI = DaFKD(dataset, device, args, mtl_model_trainer)
        fedavgAPI.train()
