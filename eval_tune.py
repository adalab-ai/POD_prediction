import os
import joblib
import json

import hydra
from sklearn.model_selection import StratifiedKFold, train_test_split
from omegaconf import open_dict
import numpy as np

import train
from tune import run_optimization, get_tune_args
from src.utils.load_data_utils import get_data
from src.utils.metrics import apply_all_metrics
from src.tune_src.tune_utils import timestring


import pandas as pd
def load_df(path, clinical, blood, sparse_img, precipitals):
    df = pd.read_csv(path).drop(columns=['POCD'])
    if not blood:
        # drop all columns that start with "blood_"
        df = df.drop(columns=df.filter(regex='^blood_'))
    if not clinical:
        # drop all columns that start with "clinical_"
        df = df.drop(columns=df.filter(regex='^clinical_'))
    if not sparse_img:
        # drop all columns that start with "sparse_img_"
        df = df.drop(columns=df.filter(regex='^sparse_img_'))
    if not precipitals:
        # drop all columns that start with "precipitals_"
        df = df.drop(columns=df.filter(regex='^precipitals_'))
    
    return df

def split_in_out(df):
    x = df.drop(columns=['POD']).to_numpy()
    y = df['POD'].to_numpy()
    feature_names = df.drop(columns=["POD"]).columns.tolist()
    return x, y, feature_names


@hydra.main(config_path="configs", config_name="eval_tune")
def main(cfg):
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    # Get path to save results
    eval_tune_folder = 'results_eval_tune'
    time_str = timestring()
    subfolder_name = f'{time_str}_{cfg.model}_{cfg.nt // 1000}k_' \
                     f'{cfg.nf_outer}_{cfg.nf_inner}_{cfg.dts}_{cfg.use_yeo_johnson}_{cfg.quantile_val}_{cfg.fill_mode}'
    path = os.path.join(eval_tune_folder, subfolder_name)
    os.makedirs(path, exist_ok=True)
    # Save eval_tune hyperparams
    joblib.dump(cfg, os.path.join(path, "cfg.pkl"))
    print("Saving to ", subfolder_name)

    # Get tune cfg and overwrite some tune settings
    with open_dict(cfg):
        nf_outer = cfg.pop('nf_outer')
        cfg['tune_model'] = cfg.pop('model')
        mode = cfg.pop("mode")
        
    tune_cfg, train_cfg = get_tune_args(override_dict=cfg)

    # Load data once to determine splits
    dts = cfg['dts']
    blood = 'blood' in dts
    clinical = 'clinical' in dts
    #imaging_pca = 'imaging_pca' in dts
    #imaging = ('imaging' in dts) and not imaging_pca
    sparse_img = 'sparse_img' in dts
    precipitals = 'precipitals' in dts

    # Load data
    data = load_df("data/merged_df.csv", clinical, blood, sparse_img, precipitals)
    x_all, y_all, feature_names = split_in_out(data)

    skf = StratifiedKFold(n_splits=nf_outer, shuffle=True)
    
    # save some test patients in JSON
    debug_dir = "debug/test_patients"
    os.makedirs(debug_dir, exist_ok=True)
    for i in range(5):
        x = x_all[i]
        input_dict = {feature_names[i]: x[i] for i in range(len(feature_names))}
        with open(os.path.join(debug_dir, f"{i}.json"), "w+") as f:
            json.dump(input_dict, f)
    
    # determine mode
    # mode 0: do not make hold-out split (do it for all other modes)
    # mode 1: tune N times
    # mode 2: tune once, then train on N different splits
    # mode 3: tune once, then train on one split N times
    tune_all = True
    if mode == 0: 
        splits = skf.split(x_all, y_all)
        x_holdout, y_holdout = None, None
    else:
        # make hold-out split
        idcs = list(range(len(x_all)))
        x_all_idcs, holdout_idcs = train_test_split(idcs, test_size=0.1, random_state=42)
        x_holdout, y_holdout = x_all[holdout_idcs], y_all[holdout_idcs]
        x_no_holdout, y_no_holdout =  x_all[x_all_idcs], y_all[x_all_idcs]
        print("All shape after holdout:", x_all.shape)
        print("Holdout shape:", x_holdout.shape)
        
        splits = list(skf.split(x_no_holdout, y_no_holdout))
        if mode == 2:
            tune_all = False
        elif mode == 3:
            splits = [splits[0] for _ in range(len(splits))]
            tune_all = False
                    
        x_all_idcs = np.array(x_all_idcs)
        splits = [(x_all_idcs[split[0]], x_all_idcs[split[1]]) for split in splits]
        
    trained_models = []
    store_lists = []
    score_dicts = []
    for split_idx, (dev_idcs, test_idcs) in enumerate(splits):
        y_dev, y_test = y_all[dev_idcs], y_all[test_idcs]
        x_dev, x_test = x_all[dev_idcs], x_all[test_idcs]
        # Run hyperparameter tuning:
        if tune_all or split_idx == 0:
            value, hyperparams, trial, best_train_args = run_optimization(x=x_dev, y=y_dev, feature_names=feature_names,
                                                                          train_args=train_cfg,
                                                                          **tune_cfg)
                                                                          
        print(value, hyperparams, best_train_args)
        # Set training hyperparams to best hyperparams found during tuning
        """
        if cfg['df'] == 'opt':
            yeo = hyperparams.pop('yeo')
            norm_method = hyperparams.pop('norm')
            fill_method = hyperparams.pop('fill')
            remove_outliers = hyperparams.pop('remove_outs')
            train_cfg.df = out_dir_name(yeo, norm_method, fill_method, remove_outliers, 0)
            train_cfg.miss_feats = hyperparams.pop('miss_feats')
        """
        for key, val in best_train_args.items():
            train_cfg[key] = val
        # Based on best hyperparams, validate on test data:
        eval_score, y_pred_logits, y_pred_binary, y_true, trained_model, preprocessor = train.start_training(x_dev, y_dev, x_test, y_test, feature_names,
                                                                                               #test_idcs=test_idcs,
                                                                                               return_preds=True,
                                                                                               **train_cfg)
        print("Test set score: ", eval_score)
        y_pred_logits, y_pred_binary, y_true = y_pred_logits[0], y_pred_binary[0], y_true[0]
        # logits from numpy arrays to lists
        y_pred_logits, y_pred_binary, y_true = y_pred_logits.tolist(), y_pred_binary.tolist(), y_true.tolist()
        score_dict = apply_all_metrics(y_true, y_pred_binary, y_pred_logits, shape_is_correct=True)

        # create folder
        os.makedirs(os.path.join(path, f'{split_idx}'), exist_ok=True)
        # store all info
        store_list = [train_cfg, eval_score, score_dict, y_pred_logits, y_pred_binary, y_true, feature_names, trained_model]
        joblib.dump(store_list, os.path.join(path, f'{split_idx}/everything.pkl'))

        def store_json(path, obj):
            with open(path, "w+") as f:
                json.dump(obj, f)
        # store cfg with joblib
        joblib.dump(cfg, os.path.join(path, f'{split_idx}/cfg.pkl'))
        # store hyperparams
        store_json(os.path.join(path, f'{split_idx}/hyperparams.json'), hyperparams)
        # store best hyperparams
        joblib.dump(best_train_args, os.path.join(path, f'{split_idx}/best_train_args.pkl'))
        # store results
        store_json(os.path.join(path, f'{split_idx}/results.json'), score_dict)
        # store predictions
        store_json(os.path.join(path, f'{split_idx}/predictions.json'), {'y_pred_logits': y_pred_logits, 'y_pred_binary': y_pred_binary, 'y_true': y_true})
        # store feature names
        store_json(os.path.join(path, f'{split_idx}/feature_names.json'), feature_names)
        # store trained model (if xgboost store it without pickle)
        import xgboost
        if isinstance(trained_model, xgboost.XGBClassifier):
            trained_model.save_model(os.path.join(path, f'{split_idx}/model.json'))
        joblib.dump(trained_model, os.path.join(path, f'{split_idx}/model.pkl'))
        # store preprocessor
        joblib.dump(preprocessor, os.path.join(path, f'{split_idx}/preprocessor.pkl'))


        store_lists.append(store_list)
        trained_models.append(trained_model)
        score_dicts.append(score_dict)

    # summarize score dicts
    score_dicts = pd.DataFrame(score_dicts)
    score_dicts.to_csv(os.path.join(path, 'score_dicts.csv'))
    # calculate mean and std of score dicts
    mean_scores = score_dicts.mean()
    std_scores = score_dicts.std()
    # put mean and std in same df
    mean_std_scores = pd.concat([mean_scores, std_scores], axis=1)
    mean_std_scores.columns = ["mean", "std"]
    # save df
    mean_std_scores.to_csv(os.path.join(path, 'mean_std_scores.csv'))

    # validate splits on holdout set
    if x_holdout is not None:
        #from src.utils.metrics import create_preds_and_reshape
        from sklearn.metrics import roc_auc_score, average_precision_score

        # get predicted logits for all models
        y_pred_logits_list = [model.predict_proba(x_holdout) for model in trained_models]
        y_pred_logits_list = [logit if len(logit.shape) == 1 else logit[:, 1] for logit in y_pred_logits_list]
        # calculate score per model
        aps = [average_precision_score(y_holdout, logits) for logits in y_pred_logits_list]
        aucs = [roc_auc_score(y_holdout, logits) for logits in y_pred_logits_list]
        mean_ap, std_ap = np.mean(aps), np.std(aps)
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        print("Individual performances:")
        print("AP: ", aps, mean_ap, std_ap)
        print("AUC: ", aucs, mean_auc, std_auc)
        # calculate score for ensemble
        mean_logits = np.mean(y_pred_logits_list, axis=0)
        ap = average_precision_score(y_holdout, mean_logits)
        auc = roc_auc_score(y_holdout, mean_logits)
        print("Ensemble performance:")
        print("AP/AUC: ", ap, auc)
        
        perf_dict = {"APs": aps, "AUCs": aucs, "mean_AP": mean_ap, "std_AP": std_ap,
                     "mean_AUC": mean_auc, "std_AUC": std_auc, "ens_AP": ap, "ens_AUC": auc}
        with open(os.path.join(path, "holdout_eval.json"), "w+") as f:
            json.dump(perf_dict, f)
        
        

if __name__ == "__main__":
    main()
