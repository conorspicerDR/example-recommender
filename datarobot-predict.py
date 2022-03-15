import requests
from credentials import *

if __name__ == '__main__':

    headers = {
        'Content-Type': '{};charset={}'.format('text/plain', 'utf-8'),
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=DEPLOYMENT_ID)

    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data='00609a1cc562140fa87a6de432bef9c9f0b936b259ad3075eb2a65008df1dbab',
        headers=headers,
    )

    print(predictions_response.text)
