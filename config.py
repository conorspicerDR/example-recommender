import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import pickle

base_path = 'data/'
csv_users = f'{base_path}customers.csv'
csv_items = f'{base_path}articles.csv'
csv_train = f'{base_path}transactions_train.csv'

# load users & items data
dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})

# Assign auto-incrementing ids starting from 0 to both users and items
ALL_USERS = dfu['customer_id'].unique().tolist()
ALL_ITEMS = dfi['article_id'].unique().tolist()
user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))
# map given IDs to 0-N IDs
user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

del dfu, dfi

if __name__ == '__main__':
    # create and save complete CSR matrix used to train/ score model
    # this is only needed as a one off run (per model version)
    df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])

    df['user_id'] = df['customer_id'].map(user_map)
    df['item_id'] = df['article_id'].map(item_map)

    # create sparse matrices
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    csr = coo.tocsr()

    save_npz("matrix.npz", csr)

    with open('user_map.pkl', 'wb') as f:
        pickle.dump(user_map, f)

    with open('item_map.pkl', 'wb') as f:
        pickle.dump(item_map, f)
