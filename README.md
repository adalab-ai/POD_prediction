# Pharmaimage

[![pipeline status](https://gitlab.com/%{project_path}/badges/%{default_branch}/pipeline.svg)](https://gitlab.com/%{project_path}/-/commits/%{default_branch})
[![coverage report](https://gitlab.com/%{project_path}/badges/%{default_branch}/coverage.svg)](https://gitlab.com/%{project_path}/-/commits/%{default_branch})

## Setting up your environment

Use `python -m pip install requirements.txt` to install the required packages.

## How to run

1. If you want to use the data for scientific research, contact Georg Winterer our paper and ask for access.
2. Move the data into a folder named "data"
3. Run the data preprocessing jupyter notebook called "prepare_data.ipynb"
4. Replicate the results of the paper by running eval_tune.py file like this: `python3 eval_tune.py -m nt=500 nf_outer=20 nf_inner=10 metric="prauc" dts=blood,sparse_img,clinical,precipitants model=xgb,log". This tunes a model for 20 folds with 500 tuning steps per fold and 10 inner split for the evaluation of a tuning step. The evaluation metric for the tuning is the prauc, the Precision Recall AUC (=average precision). The datatypes (dts) are the four individual datatypes, the script will be run separately with each of them - they can be combined, e.g. "dts=blood_clinical" to combine blood and clinical data. The chosen models are XGBoost Classifier and Logistic regression, both of which will be run after each other on each datatype.
5. The results of the nested-k-fold tuning will be stored in the folder "results_eval_tune".
6. Check the results by using the jupyter notebook at "src/eval_model_new.ipynb". Select the study names at the top of the notebook that you are interested in, you can find all study names in the "results_eval_tune" folder.
