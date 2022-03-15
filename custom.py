import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle

def transform(data):
    return data


def load_model(code_dir=''):
    with open('model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


def score_unstructured(customers, model, **kwargs):
    csr = load_npz('matrix.npz')
    with open('user_map.pkl', 'rb') as f:
        user_map = pickle.load(f)
    user_ids = {uidx: u for u, uidx in user_map.items()}

    with open('item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    item_ids = {idx: i for i, idx in item_map.items()}

    customers['user_id'] = customers['customer_id'].map(user_map)
    customers_to_score = customers[customers['user_id'].notnull()]['user_id'].unique()

    preds = []
    ids, scores = model.recommend(
        customers_to_score,
        csr[customers_to_score],
        N=12,
        filter_already_liked_items=False
    )

    for i, userid in enumerate(customers_to_score):
        customer_id = user_ids[userid]
        user_items = ids[i]
        article_ids = [item_ids[item_id] for item_id in user_items]
        preds.append((customer_id, ' '.join(article_ids)))

    df_preds = pd.DataFrame(preds, columns=['customer_id', 'prediction'])

    # join any rows not scored (unknown ID)
    return customers[['customer_id']].merge(df_preds, how='left', on=['customer_id'])


# local testing
if __name__ == '__main__':
    from datetime import datetime

    start = datetime.utcnow()
    print('Started at', start)
    customer_sample = pd.read_csv('data/customers.csv').sample(1000)

    latest_model = load_model()

    predictions = score_unstructured(customer_sample, latest_model)
    predictions.to_csv('data/local_test.csv', index=False)
    print('Total time:', str(datetime.utcnow() - start))
