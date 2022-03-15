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
        data='000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318',
        headers=headers,
    )

    print(predictions_response.text)
