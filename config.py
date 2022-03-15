import pandas as pd

base_path = 'data/'
csv_sub = f'{base_path}sample_submission.csv'
csv_users = f'{base_path}customers.csv'
csv_items = f'{base_path}articles.csv'

# load users & items data
dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})

# Assign autoincrementing ids starting from 0 to both users and items
ALL_USERS = dfu['customer_id'].unique().tolist()
ALL_ITEMS = dfi['article_id'].unique().tolist()
user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))
# map given IDs to 0-N IDs
user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

del dfu, dfi
