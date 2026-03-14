import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def fill_null_numerical_values(df, cols):
  for col in cols:
    new_column_name = col + '_missing'
    df[new_column_name] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(0)
  return df


def reformat_data(df):
    '''
    Reformating and cleaning data one last time before feeding it into the preprocessing pipeline
    Steps taken:
    1. Mapping shot_type codes to english
    2. Mapping skill level into 3 bins beginner, intermediate, advanced
    3. Force all first and second shots to be serve and return, respectively
    4. Reorder and slice dataframe
    '''
    # Step 1. mapping shot_type codes to english
    # while manually Erne and ATP to Volley and Speed Up
    mapper = {'D':'Dink', 'HB':'Volley', 'SE':'Serve', 'R':'Return',
              'tzRep':'Repel', 'tzApp':'tzApp', 'tsDrp':'Drop', 'tsDrv':'Drive',
              'SP':'Speed Up', 'Res':'Reset', 'U':'U', 'L':'Lob', 'tsLob':'Lob',
              'E':'Volley', 'A':'Speed Up', 'O':'O', 'ball':'ball'}

    df['shot_type'] = df['shot_type'].map(mapper)

    # Step 2. mapping skill level into 3 bins beginner, intermediate, advanced
    skill_mapper = {'2.5':'Beginner', '3.0':'Beginner', '3.5':'Intermediate',
                    '4.0':'Intermediate', '4.5':'Advanced',
                    '5.0':'Advanced', '5.5':'Advanced', 'Pro':'Advanced', 'Senior Pro':'Advanced'}

    df['skill_lvl'] = df['skill_lvl'].map(skill_mapper)

    # Step 3. Force all first and second shots to be serve, return, respectively
    df.loc[df.shot_nbr == 1, 'shot_type'] = 'Serve'
    df.loc[df.shot_nbr == 2, 'shot_type'] = 'Return'

    # Step 4. Reorder and slice dataframe
    df.rename(columns={'delta_x_loc':'delta_loc_x', 'delta_y_loc':'delta_loc_y'},
              inplace=True)
    new_order = ['rally_id', 'rally_nbr', 'rally_len', 'skill_lvl', 'shot_nbr',
                'shot_type', 'loc_x', 'loc_y', 'next_loc_x', 'next_loc_y',
                'delta_loc_x', 'delta_loc_y', 'shot_distance', 'shot_angle',
                'srv_point_won', 'team_hitting']
    df = df[new_order]

    #Step 5. Fill Null numerical cols
    df = fill_null_numerical_values(df, cols = ['next_loc_x', 'next_loc_y',
                'delta_loc_x', 'delta_loc_y', 'shot_distance', 'shot_angle'])

    return df

def gen_preprocess_pipe():
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_cols = ['skill_lvl']
    num_cols = ['loc_x', 'loc_y', 'next_loc_x', 'next_loc_y',
                'delta_loc_x', 'delta_loc_y', 'shot_distance', 'shot_angle', 'next_loc_x_missing']

    transformer = ColumnTransformer([
        ('cat_trans', ohe, cat_cols),
        ('num_trans', 'passthrough', num_cols)
    ])
    pipe = Pipeline([
        ('col_trans', transformer),
        ('scaler', StandardScaler())
    ])
    return pipe

def add_col_names(pipe, df):
    '''
    Called after fitting pipe
    '''
    cols = list(pipe['col_trans']['cat_trans'].get_feature_names_out())
    cols.extend(list(pipe['col_trans']['num_trans'].get_feature_names_out()))

    df = pd.DataFrame(df, columns=cols)
    return df

def create_splits(df, splits=(0.9, 0.1), seed=42):
    known_shots = ['Drop', 'Volley', 'Dink', 'Lob', 'Speed Up', 'Repel', 'Drive', 'Reset']
    known_mask = [True if x in known_shots else False for x in df.shot_type]
    unknown_shots = ['U', 'tzApp', 'ball', 'O']
    unknown_mask = [True if x in unknown_shots else False for x in df.shot_type]

    X = df[known_mask]
    y = X.shot_type
    X_unknown = df[unknown_mask]

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                    train_size=splits[0],
                                    random_state=seed)

    pipe = gen_preprocess_pipe()
    x_train = pipe.fit_transform(x_train)
    x_test = pipe.transform(x_test)

    x_train = add_col_names(pipe, x_train)
    x_test = add_col_names(pipe, x_test)

    X_unknown = pipe.transform(X_unknown)
    X_unknown = add_col_names(pipe, X_unknown)

    return x_train, x_test, y_train, y_test, X_unknown

def main(args):
    # Load data
    df = pd.read_csv(args.i_dir+'/'+args.shot_file, index_col=0)
    
    # Reformat data
    df = reformat_data(df)
    
    # create splits for training
    x_train, x_test, y_train, y_test, X_unknown = create_splits(df,
                                                                splits=(args.train_split, 1-args.train_split),
                                                                seed=0)
    # applying LabelEncoder() to y_train, y_test
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return x_train, x_test, y_train, y_test, X_unknown, label_encoder

if __name__ == '__main__':
    # writing command-line interface
    desc = """Script for preprocessing prior to training a classifier.
              Classifier is used to reduce cardinality of shot_type."""

    parser = argparse.ArgumentParser(prog='preprocessing before classifier', 
                                     description=desc)

    parser.add_argument('-i', '--i_dir', help='directory containing input file', default='./')
    parser.add_argument('-s', '--shot_file', help='name of file holding shot_rally data, should be csv',
                        default='shot_rally.csv')
    parser.add_argument('--train_split', help='train split. should be less than 1', default=0.8)
    
    args = parser.parse_args()
    args.train_split = float(args.train_split)
    
    main(args)
