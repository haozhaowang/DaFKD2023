import copy
import logging
import random
import numpy as np
import torch
from utils import transform_list_to_tensor

from client import Client

class DaFKD(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        # self.worker_num = self.args.client_num_per_round
        self.client_indexes = []
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_dict = dict()
       
        self.val_global = []
        self.train_acc = []
        self.test_acc = []
        self.probability_density_dict = dict()
        self.probability_density_dict_t = dict()

        self.model_trainer = model_trainer
       
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict,train_data_global, model_trainer)



    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, train_data_global,model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], train_data_global,self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        

        for round_idx in range(self.args.comm_round):
            w_global = self.model_trainer.get_model_params()
            
            logging.info("################Communication round : {}".format(round_idx))
            if self.args.dataset == 'cifar10':
                global_noise = torch.randn(self.args.ed_epoch,self.args.batch_size,self.args.noise_dimension,1,1)
            else:
                global_noise = torch.randn(self.args.ed_epoch,self.args.batch_size,self.args.noise_dimension)
            # global_noise = torch.randn(self.args.ed_epoch,self.args.batch_size,self.args.noise_dimension)

            self.val_global = self.model_trainer.get_distillation_share_data(global_noise,self.device)

            # self._generate_validation_set()
            w_locals = []
           
            self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(self.client_indexes))

                # update dataset
            for idx in self.client_indexes:
                client_idx = idx
                for i in self.client_list:
                    if i.client_idx == client_idx:
                        client = i

                # train on new dataset
                weight,probability_density,probability_density_t= client.train(copy.deepcopy(w_global),self.val_global,self.test_global,round_idx)
                w_locals.append((client.get_sample_number(), copy.deepcopy(weight)))

                for i in range(len(probability_density)):
                    probability_density[i] = probability_density[i].cpu().detach().tolist()

                for i in range(len(probability_density_t)):
                    probability_density_t[i] = probability_density_t[i].cpu().detach().tolist()

                self.probability_density_dict[client.client_idx] = probability_density
                self.probability_density_dict_t[client.client_idx] = probability_density_t

                self.model_dict[client.client_idx] = (transform_list_to_tensor(copy.deepcopy(weight[0])),
                                                      transform_list_to_tensor(copy.deepcopy(weight[1])),
                                                      transform_list_to_tensor(copy.deepcopy(weight[2]))                 
                                                                                                        )
            probability_density_sum = None
            for idx in self.client_indexes:
                if probability_density_sum == None:
                    probability_density_sum = copy.deepcopy(self.probability_density_dict[idx])
                else:
                    for i in range(len(self.probability_density_dict[self.client_indexes[0]])):
                        for j in range(self.args.batch_size):
                            probability_density_sum[i][j] += self.probability_density_dict[idx][i][j]

            for idx in self.client_indexes:
                for i in range(len(self.probability_density_dict[self.client_indexes[0]])):
                    for j in range(self.args.batch_size):
                        try:
                            self.probability_density_dict[idx][i][j] = self.probability_density_dict[idx][i][j] / probability_density_sum[i][j]
                        except:
                            self.probability_density_dict[idx][i][j] = 0


            probability_density_t_sum = None
            for idx in self.client_indexes:
                if probability_density_t_sum == None:
                        probability_density_t_sum = copy.deepcopy(self.probability_density_dict_t[idx])
                else:
                    for i in range(len(self.probability_density_dict_t[self.client_indexes[0]])):
                        for j in range(self.args.batch_size):
                            probability_density_t_sum[i][j] += self.probability_density_dict_t[idx][i][j]

            for idx in self.client_indexes:
                for i in range(len(self.probability_density_dict_t[self.client_indexes[0]])):  
                    for j in range(self.args.batch_size):
                        if probability_density_t_sum[i][j] == 0:
                            self.probability_density_dict_t[idx][i][j] = 0
                        else:
                            self.probability_density_dict_t[idx][i][j] = self.probability_density_dict_t[idx][i][j] / probability_density_t_sum[i][j]



            w_global = self._aggregate(w_locals,round_idx)


            if round_idx % 1 == 0:
                self._local_test_on_all_clients(round_idx)

            if round_idx % 1 == 0:
                f = open("DaFKD_mnist.txt",'w')
                for i in range(len(self.train_acc)):
                    f.write("train acc:"+str(self.train_acc[i])+" "+"test acc:"+str(self.test_acc[i])+'\n')
                f.close()


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            self.client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            self.client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # return client_indexes

    # def _generate_validation_set(self, num_samples=6000):
    #     num = num_samples // self.args.batch_size
    #     test_data_num = len(self.distillation_data)
    #     sample_indices = random.sample(range(test_data_num), min(num, test_data_num))
    #     self.val_global = torch.utils.data.Subset(self.distillation_data, sample_indices)


    def _aggregate(self, w_locals,round_idx):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, model_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, model_params) = w_locals[0]

        for k in model_params[0].keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    model_params[0][k] = local_model_params[0][k] * w
    
                else:
                    model_params[0][k] += local_model_params[0][k] * w


        for k in model_params[1].keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    model_params[1][k] = local_model_params[1][k] * w
                else:
                    model_params[1][k] += local_model_params[1][k] * w
                          
        for k in model_params[2].keys():
            for i in range(len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    model_params[2][k] = local_model_params[2][k] * w
                else:
                    model_params[2][k] += local_model_params[2][k] * w

        global_model_params = model_params

        if round_idx > 0 and round_idx < 200 :
            global_model_params = self.model_trainer.weighted_ensemble_distillation(
                self.args,self.device,
                self.val_global, self.client_indexes,
                self.model_dict, self.probability_density_dict,
                round_idx,  self.test_global, self.probability_density_dict_t, global_model_params
            )

        self.model_trainer.set_model_params(global_model_params)

        return global_model_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client in self.client_list:

            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        logging.info(stats)
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats)
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}

        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
