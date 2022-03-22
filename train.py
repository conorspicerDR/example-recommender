"""
Example of training implicit model
Based on this notebook: https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014
"""

import pickle
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix, save_npz


def to_user_item_coo(df):
    """ Turn a dataframe with transactions into a COO sparse items x users matrix """
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    return coo


def split_data(df, validation_days=7):
    " Split a pandas dataframe into training and validation data, using <<validation_days>>"
    validation_cut = df['t_dat'].max() - pd.Timedelta(validation_days)

    df_train = df[df['t_dat'] < validation_cut]
    df_val = df[df['t_dat'] >= validation_cut]
    return df_train, df_val


def get_val_matrices(df, validation_days=7):
    """ Split into training and validation and create various matrices

        Returns a dictionary with the following keys:
            coo_train: training data in COO sparse format and as (users x items)
            csr_train: training data in CSR sparse format and as (users x items)
            csr_val:  validation data in CSR sparse format and as (users x items)

    """
    df_train, df_val = split_data(df, validation_days=validation_days)
    coo_train = to_user_item_coo(df_train)
    coo_val = to_user_item_coo(df_val)

    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()

    return coo_train, csr_train, csr_val


def train(coo_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 iterations=iterations,
                                                 regularization=regularization,
                                                 random_state=42,
                                                 use_gpu=False)
    model.fit(coo_train, show_progress=show_progress)
    return model


if __name__ == '__main__':

    # Load Data
    base_path = 'data/'
    csv_train = f'{base_path}transactions_train.csv'
    csv_sub = f'{base_path}sample_submission.csv'
    csv_users = f'{base_path}customers.csv'
    csv_items = f'{base_path}articles.csv'

    df_transactions = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
    df_sub = pd.read_csv(csv_sub)
    dfu = pd.read_csv(csv_users)
    dfi = pd.read_csv(csv_items, dtype={'article_id': str})

    # Assign auto-incrementing ids starting from 0 to both users and items
    ALL_USERS = dfu['customer_id'].unique().tolist()
    ALL_ITEMS = dfi['article_id'].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))
    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df_transactions['user_id'] = df_transactions['customer_id'].map(user_map)
    df_transactions['item_id'] = df_transactions['article_id'].map(item_map)
    del dfu, dfi

    # Training
    # coo_train, csr_train, csr_val = get_val_matrices(df_transactions, 7)   # used for hyper param tuning
    best_params = {'factors': 10, 'iterations': 15, 'regularization': 0.01}  # guess at some suitable hyper-params

    # train on whole dataset
    coo = to_user_item_coo(df_transactions)
    model = train(coo, **best_params)

    # save model artifacts needed for scoring
    artifacts_path = 'artifacts/'
    csr = coo.tocsr()
    save_npz(f'{artifacts_path}matrix.npz', csr)

    with open(f'{artifacts_path}user_map.pkl', 'wb') as f:
        pickle.dump(user_map, f)

    with open(f'{artifacts_path}item_map.pkl', 'wb') as f:
        pickle.dump(item_map, f)

    with open(f'{artifacts_path}model.sav', 'wb') as pickle_out:
        pickle.dump(model, pickle_out)
