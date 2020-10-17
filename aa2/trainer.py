import os
import json
from collections import defaultdict
from itertools import product
import time

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from .trainer_helper import buid_model, train_epoch, evaluate_epoch, calculate_time
from .early_stopping import EarlyStopping


class Trainer:

    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)

    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name, paramaters):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc
        #
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-model_dict-for-inference-and-or-resuming-training
        hyperparamaters["epoch"] = epoch
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparamaters': hyperparamaters,
            'loss': loss,
            'scores': scores,
            'model_name': model_name,
            'parameters': paramaters,
        }

        torch.save(save_dict, os.path.join(
            self.dump_folder, model_name + ".pt"))

    def load_model(self, model_path):
        # Finish this function so that it loads a model and return the appropriate variables
        model_dict = torch.load(model_path)

        return model_dict

    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
        # get some variables
        device = train_X.device
        set_glove_v = hyperparamaters["embeddings"]
        # Construct Tensordatasets
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)

        # get the variable hyperparamaters
        para_var_id = []
        para_fixed_id = []
        hyperpara_name = list(hyperparamaters.keys())
        for name, val in hyperparamaters.items():
            if type(val) == "dict":
                val = val.keys()

            if len(val) > 1:
                para_var_id.append(hyperpara_name.index(name))
            else:
                para_fixed_id.append(hyperpara_name.index(name))

        # all possible combinations of hyperparamaters
        set_paramaters = product(*hyperparamaters.values())
        # somedata = defaultdict(list)

        # Parameter index in paramaters
        #   0:  "learning_rate"
        #   1:  "number_layers"
        #   2:  "optimizer",
        #   3:  "bidirection"
        #   4:  "dropout"
        #   5:  "epochs"
        #   6:  "batch_size"
        #   7:  "embeddings"
        #   8:  "num_nes" = "outpit_size"
        #   9:  "pad"

        # defining models
        print("Model Training ...")
        for j, paramaters in enumerate(set_paramaters):
            # setting up hyperparameter
            lr = paramaters[0]
            num_layers = paramaters[1]
            opt = paramaters[2]
            num_dirs = paramaters[3]
            dropout_ = paramaters[4]
            epochs = paramaters[5]
            batch_size = paramaters[6]
            embd_type = paramaters[7]
            out_size = paramaters[8]
            pad_token_id = paramaters[9]

            # get the variable and fixed hyperparamaters names
            para_var_name = [hyperpara_name[i] for i in para_var_id]

            # get the variable and fixed hyperparamaters values
            # convert non numeric values like str to numeric if they are variable hyperparamaters as ariable hyperparamaters are plotted in parallel coordination plot.
            # variable paramaters are used to construct model name, so we convert the numeric values to str
            para_var = []
            mdl_name = []  
            for i in para_var_id:
                if isinstance(paramaters[i], int) or isinstance(paramaters[i], float):
                    para_var.append(paramaters[i])
                    mdl_name.append(str(paramaters[i]))
                else:
                    name_ = hyperpara_name[i]
                    all_values = hyperparamaters[name_]
                    if type(all_values) == dict:
                        para_var.append(
                            list(all_values.keys()).index(paramaters[i]))
                        mdl_name.append(".".join(map(str, paramaters[i])))
                    else:
                        para_var.append(all_values.index(paramaters[i]))
                        mdl_name.append(paramaters[i])

            # construct model name
            mdl_name = "_".join(mdl_name)

            # Construct batcher
            train_iter = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True
                                    )
            val_iter = DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True
                                  )

            # build model
            # define sizes
            pretrained_embed = set_glove_v[embd_type]
            input_size = pretrained_embed.shape[0]
            embed_layer_out_size = pretrained_embed.shape[1]
            lstm_hidden_size = (embed_layer_out_size // 3) + 1
            model_sizes = [ input_size,
                            embed_layer_out_size,
                            lstm_hidden_size,
                            num_layers,
                            num_dirs,
                            dropout_,
                            out_size
                        ]
            mdl = buid_model(model_sizes, model_class)
            mdl.embedding.weight.data.copy_(pretrained_embed)

            # train and validate the model
            criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            if opt == "adam":
                optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(mdl.parameters(), lr=lr)

            mdl.to(device)
            criterion.to(device)
            
            es = EarlyStopping(patience=5)  # early stopping
            best_f1_val = -1    # metric score to get the best model
            start_time = time.time()    # time
            for epoch in range(epochs):
                # train, evaluate
                _loss_trn, _f1_trn = train_epoch(
                    mdl, train_iter, optimizer, criterion, pad_token_id)
                loss_val, _loss_val_value, f1_val = evaluate_epoch(
                    mdl, val_iter, criterion, pad_token_id)

                # instead of saving all models save the best one
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    self.save_model(
                        epoch,
                        mdl,
                        optimizer,
                        loss_val,
                        {"F1_score": f1_val},
                        # variable paramaters to be polt
                        {k: v for k, v in zip(para_var_name, para_var)},
                        mdl_name,
                        # all paramaters
                        {k: v for k, v in zip(hyperpara_name, paramaters)},
                    )

                if es.step(_loss_val_value):
                    break
            
            # get running time
            end_time = time.time()
            mins, secs = calculate_time(start_time, end_time)
            
            print(
                "model-{:02d}: {: <15s} |   Best score: {:.3f}   | time: {: >3d} min and {: >3d} sec".format(j, mdl_name, best_f1_val, mins, secs))

            mdl = mdl.cpu()

        # with open("/home/guszarzmo@GU.GU.SE/LT2316-H20/Assignments/lt2316-h20-aa2/data/somedata.json", "w") as fout:
        #     json.dump(somedata, fout)

    def test(self, test_X, test_y, model_class, best_model_path, pad_token_id):
        # Finish this function so that it loads a model, test is and print results.
        model_dict = self.load_model(best_model_path)
        model_state = model_dict['model_state_dict']
        # optimizer_state = model_dict['optimizer_state_dict']
        model_paramaters = model_dict['parameters']
        
        batch_size = model_paramaters["batch_size"]
        num_layers = model_paramaters["number_layers"]
        num_dirs = model_paramaters["bidirection"]
        input_size = model_state["embedding.weight"].shape[0]
        embed_layer_out_size = model_state["embedding.weight"].shape[1]
        lstm_hidden_size = model_state["fc.weight"].shape[1] // 2
        out_size = model_state["fc.weight"].shape[0]
        dropout_ = model_paramaters["dropout"]
        
        model_sizes = [ input_size,
                        embed_layer_out_size,
                        lstm_hidden_size,
                        num_layers,
                        num_dirs,
                        dropout_,
                        out_size
                    ]
        model = buid_model(model_sizes, model_class)
        device = test_X.device
        model.to(device)
        # load model state to the new constructed model
        model.load_state_dict(model_state)
        for parameter in model.parameters():
            parameter.requires_grad = False

        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        # Construct Tensordatasets
        test_dataset = TensorDataset(test_X, test_y)
        # Construct batcher
        test_iter = DataLoader(test_dataset,
                               batch_size=batch_size,
                               shuffle=True
                               )
        _, _, f1_val = evaluate_epoch(
            model, test_iter, criterion, pad_token_id)

        return f1_val
