import os
import logging
import traceback
import torch
import numpy as np

def transform_list_to_tensor(model_params_list):
    try:
        for k in model_params_list.keys():
            model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
        return model_params_list
    except Exception as e:
        logging.error(traceback.format_exc())


def transform_tensor_to_list(model_params):
    try:
        for k in model_params.keys():
            model_params[k] = model_params[k].detach().numpy().tolist()
        return model_params
    except Exception as e:
        logging.error(traceback.format_exc())