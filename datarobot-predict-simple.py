import pandas as pd
import requests

# THESE NEED TO BE UPDATED
# They can be found on Deployment > Predictions > Prediction API
DEPLOYMENT_ID = '123456789'
API_URL = 'https://mlops.dynamic.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictionsUnstructured'
API_KEY = 'MY_API_KEY'
DATAROBOT_KEY = 'KEY'

headers = {
    'Content-Type': '{};charset={}'.format('text/plain', 'utf-8'),
    'Authorization': 'Bearer {}'.format(API_KEY),
    'DataRobot-Key': DATAROBOT_KEY,
}
url = API_URL.format(deployment_id=DEPLOYMENT_ID)


def get_predictions(cust_id):
    predictions_response = requests.post(
        url,
        data=cust_id,
        headers=headers,
    )
    return predictions_response.content


if __name__ == '__main__':

    from datetime import datetime
    start = datetime.utcnow()
    print('Starting at:', str(start))

    df = pd.read_csv('data/customers_1000.csv')
    df['predictions'] = df['customer_id'].apply(get_predictions)
    df.to_csv('data/simple_preds.csv', index=False)

    print('Finished at:', str(start))
    print('Total time:', str(datetime.utcnow() - start))
