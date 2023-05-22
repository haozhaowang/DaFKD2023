import logging
import copy
class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, train_data_global,args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.train_data_global = train_data_global
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.model_d_para = None
        self.time = 0

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,w_global,distillation_share_data,test_global,round_idx):
        # self.time += 1 
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args,round_idx)
        # if self.time == 3:
        #     self.model_d_para = self.model_trainer.get_model_params_all()

        weights = self.model_trainer.get_model_params()
        # if self.model_d_para:
        #     temp = self.model_trainer.get_model_params_all()
        #     self.model_trainer.set_model_params_all(copy.deepcopy(self.model_d_para))
        #     probability_density_d = self.model_trainer.get_probability_density(distillation_share_data,self.device)
        #     probability_density_t = self.model_trainer.get_probability_density(test_global,self.device)
        #     self.model_trainer.set_model_params_all(temp)
        # else:
        probability_density_d = self.model_trainer.get_probability_density(distillation_share_data,self.device)
        probability_density_t = self.model_trainer.get_probability_density(test_global,self.device)

        return weights,probability_density_d,probability_density_t

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
