import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from datetime import datetime

base_path = 'data/'
csv_train = f'{base_path}transactions_train.csv'
csv_sub = f'{base_path}sample_submission.csv'
csv_users = f'{base_path}customers.csv'
csv_items = f'{base_path}articles.csv'

df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
df_sub = pd.read_csv(csv_sub)
dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})

ALL_USERS = dfu['customer_id'].unique().tolist()
ALL_ITEMS = dfi['article_id'].unique().tolist()

user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))

# map given IDs to 0-N IDs
user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

df['user_id'] = df['customer_id'].map(user_map)
df['item_id'] = df['article_id'].map(item_map)

row = df['user_id'].values
col = df['item_id'].values
data = np.ones(df.shape[0])

del df, dfu, dfi, user_map
coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
csr_train = coo_train.tocsr()


def transform(data):
    """ Turn a dataframe with transactions into a COO sparse items x users matrix"""
    return data


def load_model():
    with open('model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


def score(data, model, **kwargs):
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USERS))
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx: startidx + batch_size]
        ids, scores = model.recommend(batch, data[batch], N=12, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = user_ids[userid]
            user_items = ids[i]
            article_ids = [item_ids[item_id] for item_id in user_items]
            preds.append((customer_id, ' '.join(article_ids)))

    predictions = pd.DataFrame(preds, columns=['customer_id', 'prediction'])

    return predictions


if __name__ == '__main__':
    start = datetime.utcnow()
    latest_model = load_model()

    df_preds = score(csr_train, latest_model)
    df_preds.to_csv('data/local_test.csv', index=False)
    print('Total time:'(datetime.utcnow() - start).total_seconds())

