import os
import argparse
import joblib
import json

import numpy as np
import shap
import pandas as pd
from scipy.stats import yeojohnson


def load_models(path):
    model_list = [f for f in os.listdir(path) if f.endswith(".pkl") and not f.startswith("cfg.")]
    loaded_data = [joblib.load(os.path.join(path, model_name)) for model_name in model_list]
    loaded_models = [d[-1] for d in loaded_data]  # last element is the model
    feature_names = loaded_data[0][-2]  # feature_names are the same for all models of that study, is second to last entry
    cfg = loaded_data[0][0]  # cfg is the same for all models of that study, is first entry
    return cfg, loaded_models, feature_names


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", help="Path to trained cross-validation model")
    argparser.add_argument("--input_path", help="Path to input JSON")
    argparser.add_argument("--output_path", help="Path to output JSON", default="outputs/test.json")
    argparser.add_argument("--verbose", default=1, type=int)
    argparser.add_argument("--preprocess", default=False, action="store_true")
    args = vars(argparser.parse_args())
    model_path = args["model_path"]
    input_path = args["input_path"]
    output_path = args["output_path"]
    verbose = args["verbose"]
    
    # load models
    cfg, loaded_models, feature_names = load_models(model_path)
    if verbose:
        print("Num loaded models: ", len(loaded_models))
        print("Loaded order of feature names: ", feature_names)
        print("Num features: ", len(feature_names))
    
    # load input
    with open(input_path, "r") as f:
        model_input_dict = json.load(f)

    # bring input into correct shape and order
    model_input = np.array([[model_input_dict[name] for name in feature_names if name in model_input_dict]])
    if verbose:
        print("Loaded input shape: ", model_input.shape)

    # preprocess input
    if args["preprocess"]:
        # load mean and std
        means = pd.read_csv("data/norm_means.csv", index_col=0).loc[feature_names].to_numpy().reshape(-1)
        stds = pd.read_csv("data/norm_stds.csv", index_col=0).loc[feature_names].to_numpy().reshape(-1)
        # load lambdas
        lambdas = pd.read_csv("data/lambdas.csv", index_col=0).loc[feature_names].to_numpy().reshape(-1)
        # load clipping thresholds
        thresholds = pd.read_csv("data/clipping_thresholds.csv", index_col=0)

        # apply yeo-johnson transform
        if "Yeo_Y" in cfg["df"]:
            # apply yeo-johnson transform with own lambda per feature
            model_input = [yeojohnson(val, lmbda=lmbda) for val, lmbda in zip(model_input, lambdas)]
            model_input = np.array([model_input])
        
        # apply normalization
        if "/z/" in cfg["df"]:
            model_input = (model_input - means) / stds
        # Normalization:
    df = normalize(df, method=args.norm_method)

    # Fill NaNs:
    df = fill_missings(df, args.fill_method, args.estimator)

    # Outlier detection & handling per dataframe
    df, new_dev_idcs = _remove_outliers(df, dev_idcs, test_idcs, path, args)


    
    # make predictions
    proba_predictions = [float(model.predict_proba(model_input)[0][1]) for model in loaded_models]
    binary_predictions = [p > 0.5 for p in proba_predictions]
    mean_pred_proba = float(np.mean(proba_predictions))
    mean_pred_binary = mean_pred_proba > 0.5
    proba_std = float(np.std(proba_predictions))

    # get shapley values/feature importances
    shap_vals = []
    for model in loaded_models:
        explainer = shap.TreeExplainer(model)
        shap_vals.append(explainer.shap_values(model_input)[0])
    shap_vals = np.array(shap_vals)
    shap_vals_mean = np.mean(shap_vals, axis=0)
    shap_vals_std = np.std(shap_vals, axis=0)
    
    # save predictions to json
    out_dict = {"proba_predictions": proba_predictions, "binary_predictions": binary_predictions,
                "mean_pred_proba": mean_pred_proba, "mean_pred_binary": mean_pred_binary,
                "proba_std": proba_std, 
                }
    #out_dict["shap_vals"] = shap_vals.tolist()
    out_dict["shap_vals_mean"] = shap_vals_mean.tolist()
    out_dict["shap_vals_std"] = shap_vals_std.tolist()
    out_dict["feature_names"] = feature_names
    with open(output_path, "w+") as f:
        json.dump(out_dict, f)
    
    if verbose:
        print("Saved predictions to ", output_path)
    
    
if __name__ == "__main__":
    main()
