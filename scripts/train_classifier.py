import numpy as np
import pandas as pd
import xgboost as xgb
import argparse
from scripts import preprocessing

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def main(args):
    # preprocess using preprocessing.py
    x_train, x_test, y_train, y_test, X_unknown, labelencoder = preprocessing.main(args)
    # declare model
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',
        device="cuda",
        objective='multi:softmax',
        num_class=10,
        eval_metric='mlogloss',
        random_state=42
    )
    # declare grid search parameters
    param_grid = {
        'n_estimators': [600, 650, 700],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }

    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=skf,
        verbose=1
    )
    # fit using grid search
    grid_search.fit(x_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.4f}")

    best_xgb = grid_search.best_estimator_
    
    # Save model in JSON format
    best_xgb.save_model(args.o_dir + '/' + args.model_file)

    return x_test, y_test, X_unknown, labelencoder

if __name__ == '__main__':
    # writing command-line interface
    desc = """Script for a shot_type classifier. Uses XGBoost."""

    parser = argparse.ArgumentParser(prog='shot_type classifier using XGBoost', 
                                     description=desc)

    parser.add_argument('-i', '--i_dir', help='directory containing input file', default='./')
    parser.add_argument('-o', '--o_dir', help='output directory containing model file', default='./')
    parser.add_argument('-s', '--shot_file', help='name of file holding shot_rally data, should be csv',
                        default='shot_rally.csv')
    parser.add_argument('-m', '--model_file', help='name of model file', default='model.json')
    parser.add_argument('--train_split', help='train split. should be less than 1', default=0.8)
    
    args = parser.parse_args()
    args.train_split = float(args.train_split)

    main(args)