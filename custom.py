# https://github.com/datarobot/datarobot-user-models/blob/master/DEFINE-INFERENCE-MODEL.md#unstructured_inference_model

from scipy.sparse import load_npz
import pickle


def load_model(code_dir=''):
    with open('model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)

    csr = load_npz('matrix.npz')

    # user ID lookup
    with open('user_map.pkl', 'rb') as f:
        user_map = pickle.load(f)

    # item ID lookup
    with open('item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    item_ids = {idx: i for i, idx in item_map.items()}

    return {'model': model, 'csr': csr, 'user_map': user_map, 'item_ids': item_ids,}


def score_unstructured(m, customer_id, **kwargs):

    user_id = m['user_map'].get(customer_id)
    # if csr is empty for a user, but their ID has been seen, model will always predict items with index 0-11
    if user_id:
        ids, scores = m['model'].recommend(
            user_id,
            m['csr'][user_id],
            N=12,
            filter_already_liked_items=True
        )
        article_ids = [m['item_ids'][item_id] for item_id in ids]
    else:
        article_ids = []

    return str({
        'customer_id': customer_id,
        'prediction': ' '.join(article_ids)
    })


# local testing
# if __name__ == '__main__':
#     from datetime import datetime
#
#     start = datetime.utcnow()
#     print('Started at', start)
#     customer_sample = '000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318'
#
#     latest_model = load_model()
#
#     prediction = score_unstructured(latest_model, customer_sample)
#     print(prediction)
#     print('Total time:', str(datetime.utcnow() - start))
