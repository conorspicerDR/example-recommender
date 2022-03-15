"""
Usage:
    python datarobot-predict.py <input-file> [mimetype] [charset]

This example uses the requests library which you can install with:
    pip install requests
We highly recommend that you update SSL certificates with:
    pip install -U urllib3[secure] certifi
"""
import sys
import json
import requests

# moved API key etc. into file not tracked by git
from credentials import *

# Don't change this. It is enforced server-side too.
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset):
    """
    Make unstructured predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs/predictions/api/dr-predapi.html

    Parameters
    ----------
    data : bytes
        Bytes data read from provided file.
    deployment_id : str
        The ID of the deployment to make predictions with.
    mimetype : str
        Mimetype describing data being sent.
        If mimetype starts with 'text/' or equal to 'application/json',
        data will be decoded with provided or default(UTF-8) charset
        and passed into the 'score_unstructured' hook implemented in custom.py provided with the model.

        In case of other mimetype values data is treated as binary and passed without decoding.
    charset : str
        Charset should match the contents of the file, if file is text.

    Returns
    -------
    data : bytes
        Arbitrary data returned by unstructured model.


    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        'Content-Type': '{};charset={}'.format(mimetype, charset),
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return raw response content
    return predictions_response.content


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(
            code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)


def main(filename, deployment_id, mimetype, charset):
    """
    Return an exit code on script completion or error. Codes > 0 are errors to the shell.
    Also useful as a usage demonstration of
    `make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset)`
    """
    data = open(filename, 'rb').read()
    data_size = sys.getsizeof(data)
    if data_size >= MAX_PREDICTION_FILE_SIZE_BYTES:
        print((
            'Input file is too large: {} bytes. '
            'Max allowed size is: {} bytes.'
        ).format(data_size, MAX_PREDICTION_FILE_SIZE_BYTES))
        return 1
    try:
        predictions = make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset)
    except DataRobotPredictionError as exc:
        print(exc)
        return 1
    print(predictions)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Input file is required argument. '
            'Usage: python datarobot-predict.py <input-file> [mimetype] [charset]')
        exit(1)
    filename = sys.argv[1]
    mimetype = sys.argv[2] if len(sys.argv) >= 3 else 'text/plain'
    charset = sys.argv[3] if len(sys.argv) == 4 else 'UTF-8'
    sys.exit(main(filename, DEPLOYMENT_ID, mimetype, charset))