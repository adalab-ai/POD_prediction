#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from pickletools import optimize

from sklearn.preprocessing import PowerTransformer, StandardScaler
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import argparse
import time
import pickle
import ast

import hydra
from hydra import compose, initialize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import mlflow
from mlflow import log_param, set_experiment
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from src.torch_src.torch_sklearn_wrapper import SklearnLightning
from src.utils.load_data_utils import get_data
from src.utils.logging import log_time
from src.eval import evaluate_model
from src.utils.metrics import ap_score, auc_score, acc_score, mean_confidence_interval, f1_score, \
    create_preds_and_reshape


def _get_class_weights(m, class_weight_tensor, pos_weight):
    if m == 'mlp':
        print("WARNING - Class weights not available for " + str(m))
        return None
    if pos_weight:
        if m == 'torch':
            class_weights = torch.tensor([1, pos_weight]).float()
        else:
            class_weights = {0: 1, 1: pos_weight}
    else:
        if m == 'torch':
            class_weights = class_weight_tensor.float()
        else:
            class_weights = "balanced"
    return class_weights


def _init_model(m, feature_names, class_weight_tensor, callbacks, use_class_weights=False, v=0, pos_weight=1,
                **model_kwargs):
    # Set class weights if used (not available for MLP and XGBoos - in that case returns None)
    class_weights = None
    if use_class_weights:
        class_weights = _get_class_weights(m, class_weight_tensor, pos_weight)
        if v:
            print("Class weights: ", class_weights)

    if m == "log":
        model = LogisticRegression(class_weight=class_weights,
                                   **model_kwargs)
    elif m == 'rf':
        model = RandomForestClassifier(class_weight=class_weights,
                                       **model_kwargs, )
    elif m == 'xgb':
        model = xgb.XGBClassifier(n_jobs=1,
                                  class_weights=class_weights,
                                  eval_metric="logloss",
                                  use_label_encoder=False,
                                  #tree_method="gpu_hist",
                                  **model_kwargs)
    elif m == 'svc':
        model = SVC(class_weight=class_weights,
                    probability=True,
                    **model_kwargs)
    elif m == 'mlp':
        hidden_layer_sizes = model_kwargs.pop("hidden_layer_sizes")
        # convert string e.g. (100, 100) to tuple of ints
        hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes[1:-1].split(",") if len(size) > 0)
        model = MLPClassifier(verbose=0,
                              hidden_layer_sizes=hidden_layer_sizes,
                              **model_kwargs)

    elif m == "torch":
        logger = None
        # mlf_logger.log_hyperparams(args)
        num_classes = 1
        model = SklearnLightning(feature_names,
                                 num_classes, logger, class_weights, callbacks,
                                 v=v,
                                 **model_kwargs)
    elif m == "tabnet":
        model = TabNetClassifier(verbose=0,
                                 **model_kwargs)
    else:
        raise NotImplementedError("Unknown model: " + str(m))
    return model


def _calc_eval_score(models, y_eval, x_eval, score_fnc_name):
    """Makes predictions for all models and returns the mean score and the confidence interval"""
    # Choose function for eval score
    if score_fnc_name == 'rocauc':
        score_fnc = auc_score
    elif score_fnc_name == 'prauc':
        score_fnc = ap_score
    elif score_fnc_name == 'acc':
        score_fnc = acc_score
    elif score_fnc_name == 'f1':
        score_fnc = f1_score
    else:
        raise NotImplementedError("Unknown score function name: " + score_fnc_name)

    preds = [score_fnc(models[i], x_eval[i], y_eval[i]) for i in range(len(models))]
    if len(preds) > 1:
        mean, low, high = mean_confidence_interval(preds, confidence=0.95)
    else:
        mean, low = preds[0], preds[0]
        high = None
    return mean, low, high


def _train_model(m, feature_names, x_train, y_train, x_eval, y_eval, class_weights, callbacks, use_class_weights=False,
                 pos_weight=1, v=0, **model_args):
    """Trains either a single model or a list of models if k-fold is selected"""
    models = []
    for i in range(len(x_train)):
        model = _init_model(m, feature_names, class_weights, callbacks, use_class_weights=use_class_weights, v=v,
                            pos_weight=pos_weight, **model_args)
        # The torch model trains using the eval split as well.
        if m == "torch":
            model.fit(x_train[i], y_train[i], x_eval[i], y_eval[i])
        elif m == "tabnet":
            model.fit(x_train[i], y_train[i], eval_set=[(x_eval[i], y_eval[i])])
        else:
            model.fit(x_train[i], y_train[i])
        models.append(model)
    return models


class Preprocesser:
        """ Preprocessor class to first clip upper and lower quantiles, then potentially fill in values either with the median, 
        then apply the PowerTransform. All quantiles, medians, etc are calculated based on the train data
        """
        def __init__(self, quantile_val, fill_mode, use_yeo_johnson, v):
            self.quantile_val = quantile_val
            self.fill_mode = fill_mode
            self.use_yeo_johnson = use_yeo_johnson
            self.v = v
            self.quantiles = None
            self.medians = None
            if use_yeo_johnson:
                self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                self.transformer = StandardScaler()
        
        def fit(self, x_train):
            # Calculate the upper and lower quantiles for each feature and the median for each class, ignoring NaNs
            self.quantiles = np.nanpercentile(x_train, (1 - self.quantile_val) * 100, axis=0), np.nanpercentile(x_train, self.quantile_val * 100, axis=0)
            self.medians = np.nanmedian(x_train, axis=0)
            # fit the transformer
            self.transformer.fit(x_train)

            if self.v:
                print('Quantiles: ', self.quantiles)
                print('Medians: ', self.medians)
                if self.use_yeo_johnson:
                    print("Lambdas: ", self.transformer.lambdas_)
            return self
        
        def transform(self, x):
            # clip the upper and lower quantiles
            x_transformed = np.clip(x, self.quantiles[0], self.quantiles[1])
            # fill in the values with the median
            if self.fill_mode == 'median':
                x_transformed = np.where(np.isnan(x_transformed), self.medians, x_transformed)
            # apply the transformation
            x_transformed = self.transformer.transform(x_transformed)
            return x_transformed

        def fit_transform(self, x):
            return self.fit(x).transform(x)


def start_training(x_train, y_train, x_eval, y_eval, feature_names,
                   v=0, m=None, model_args=None, callbacks=None, score_fnc_name='rocauc',
                   return_preds=False,
                   use_class_weights=False, split=None, nf=0,
                   save=False,
                   features=None, pos_weight=1,
                   quantile_val=0.9999, fill_mode='median', use_yeo_johnson=True,
                   **kwargs,
                   ):
    """Method to train and save supported models.
    (Called by hyperparam optimizer.)
    """
    if callbacks is None:
        use_optuna = False
        callbacks = []
    else:
        use_optuna = True
    torch.set_num_threads(1)

    # Populate training and testing data structures
    """
    x_train, y_train, x_eval, y_eval, n_features, feature_names, class_weights = \
        get_data(df_name=df, split=split, nf=nf, v=v, blood=blood, static=static, clinical=clinical, imaging=imaging,
                 imaging_pca=imaging_pca, miss_feats=miss_feats, imaging_pca_var=imaging_pca_var,
                 features=features, sparse_img=sparse_img,
                 dev_idcs=dev_idcs, test_idcs=test_idcs,
                 train_idcs=train_idcs, val_idcs=val_idcs)
    """

    if v > 0:
        print("Training shape in train: ", x_train[0].shape)
        print('Starting training... ')

    
    # normalize features
    if v > 0:
        print('Normalizing features...')
    preprocessor = Preprocesser(quantile_val=quantile_val, fill_mode=fill_mode, use_yeo_johnson=use_yeo_johnson, v=v)
    x_train = preprocessor.fit_transform(x_train)
    x_eval = preprocessor.transform(x_eval)
    # wrap list around all data
    x_train = [x_train]
    x_eval = [x_eval]
    y_train = [y_train]
    y_eval = [y_eval]


    # Train
    pos_class_frac = y_train[0].mean()
    neg_class_frac = 1 - pos_class_frac
    pos_weight = neg_class_frac / pos_class_frac
    class_weights = torch.tensor([1, pos_weight])
    models = _train_model(m, feature_names, x_train, y_train, x_eval, y_eval, class_weights, callbacks,
                          use_class_weights=use_class_weights, pos_weight=pos_weight, v=v, **model_args)

    # Calculate baseline accuracy
    calc_y_train = np.concatenate(y_train).flatten()
    baseline_acc = calc_y_train.mean()
    if baseline_acc < 0.5:
        baseline_acc = 1 - baseline_acc
    if v > 0:
        print(f'\nBaseline accuracy:\n  {baseline_acc:0.3f}')

    def create_eval_string(name, mean, low, high):
        return f'Mean {name}:\n  {low:0.3f}' + (f' - {mean:.3f} - {high:.3f}' if high is not None else '')

    # training score
    train_acc, train_acc_low, train_acc_high = _calc_eval_score(models, y_train, x_train, "acc")
    train_rocauc, train_rocauc_low, train_rocauc_high = _calc_eval_score(models, y_train, x_train, "rocauc")
    train_prauc, train_prauc_low, train_prauc_high = _calc_eval_score(models, y_train, x_train, "prauc")

    if v > 0:
        print(create_eval_string('train accuracy', train_acc, train_acc_low, train_acc_high))
        print(create_eval_string('train roc AUC', train_rocauc, train_rocauc_low, train_rocauc_high))
        print(create_eval_string('train pr AUC', train_prauc, train_prauc_low, train_prauc_high))

    timestamp = time.strftime("%Y-%model_type-%d-%H-%M-%S")
    # Validation score (only if data was split)
    eval_score = None
    y_pred_logits, y_pred_binary, y_true = None, None, None
    if split != 'no-split':
        preds = [create_preds_and_reshape(model, x_eval[idx], y_eval[idx]) for idx, model in enumerate(models)]
        y_pred_logits = [pred[0] for pred in preds]
        y_pred_binary = [pred[1] for pred in preds]
        y_true = [pred[2] for pred in preds]

        eval_f1, eval_f1_low, eval_f1_high = _calc_eval_score(models, y_eval, x_eval, "f1")
        eval_rocauc, eval_rocauc_low, eval_rocauc_high = _calc_eval_score(models, y_eval, x_eval, "rocauc")
        eval_acc, eval_acc_low, eval_acc_high = _calc_eval_score(models, y_eval, x_eval, "acc")
        eval_prauc, eval_prauc_low, eval_prauc_high = _calc_eval_score(models, y_eval, x_eval, "prauc")
        if score_fnc_name == "rocauc":
            eval_score = eval_rocauc
        elif score_fnc_name == "prauc":
            eval_score = eval_prauc
        else:
            eval_score = eval_acc

        if v > 0:
            print(create_eval_string('val AUC', eval_rocauc, eval_rocauc_low, eval_rocauc_high))
            print(create_eval_string('val acc', eval_acc, eval_acc_low, eval_acc_high))
            print(create_eval_string('val F1', eval_f1, eval_f1_low, eval_f1_high))
            print(create_eval_string('val AP', eval_prauc, eval_prauc_low, eval_prauc_high))
        # Only evaluate if we are not hyperparameter tuning
        if not use_optuna:
            # Evaluate all models
            eval_results_path = os.path.join('train_val_results', timestamp + '-' + type(models[0]).__name__)
            os.makedirs(eval_results_path, exist_ok=True)
            evaluate_model(models, x_eval, y_eval, eval_results_path, feature_names)
    # Saving
    if save:
        dirname = '/models/{}-{}'.format(timestamp, type(models[0]).__name__)
        os.makedirs(dirname, exist_ok=True)  # create a directory [timestamp + model name]
        # if k-fold save in one folder
        if nf:
            for i in range(nf):
                pickle.dump(models[i], open('{}/model{}.pkl'.format(dirname, i), 'wb'))
        else:
            pickle.dump(models[0], open('{}/model.pkl'.format(dirname), 'wb'))
        if v > 0:
            print('\nModel and parameters are saved in [{}]'.format(dirname))

    mlflow.end_run()

    if return_preds:
        return eval_score, y_pred_logits, y_pred_binary, y_true, models[0], preprocessor
    else:
        return eval_score, models


def get_args(override_dict: dict = None):
    override_list = []
    if override_dict is not None:
        for key, value in override_dict.items():
            if value is None:
                value = ""
            override_list.append(f'{key}={value}')
    cfg = compose(config_name="train", overrides=override_list)
    return cfg


def get_args_and_train():
    cfg = get_args()
    print("Args: ", cfg)
    out = start_training(**cfg)
    return out


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    print("Args: ", cfg)
    start_training(**cfg)
    return True


if __name__ == '__main__':
    # args = get_args()
    main()
