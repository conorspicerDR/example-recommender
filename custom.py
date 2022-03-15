import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from config import user_map, item_map, ALL_USERS, ALL_ITEMS, user_ids, item_ids


def transform(data):
    """ Turn a dataframe with customer IDs into a COO sparse users x items matrix"""
    data['user_id'] = data['customer_id'].map(user_map)
    data['item_id'] = data['article_id'].map(item_map)

    # create sparse matrices
    row = data['user_id'].values
    col = data['item_id'].values
    data = np.ones(data.shape[0])
    coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    csr_train = coo_train.tocsr()

    return csr_train


def load_model(code_dir=''):
    with open('model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


def score_unstructured(data, model, **kwargs):
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


# # local testing
# if __name__ == '__main__':
#     from datetime import datetime
#
#     start = datetime.utcnow()
#     print('Started at', start)
#
#     base_path = 'data/'
#     csv_train = f'{base_path}transactions_train.csv'
#     transactions_df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
#
#     prediction_cut = transactions_df['t_dat'].max() - pd.Timedelta(days=30)
#     transactions_pred = transactions_df[transactions_df['t_dat'] > prediction_cut].copy()
#     del transactions_df
#     csr_pred = transform(transactions_pred)
#
#     latest_model = load_model()
#
#     df_preds = score(csr_pred, latest_model)
#     df_preds.to_csv('data/local_test.csv', index=False)
#     print('Total time:', str(datetime.utcnow() - start))
