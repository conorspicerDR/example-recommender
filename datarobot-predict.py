import pandas as pd
from credentials import *

import asyncio
import ast
from aiohttp import ClientSession, ClientConnectorError

headers = {
    'Content-Type': '{};charset={}'.format('text/plain', 'utf-8'),
    'Authorization': 'Bearer {}'.format(API_KEY),
    'DataRobot-Key': DATAROBOT_KEY,
}
url = API_URL.format(deployment_id=DEPLOYMENT_ID)


async def make_prediction_request(customer_id: str, session: ClientSession, **kwargs) -> dict:
    try:
        resp = await session.request(method="POST", url=url, headers=headers, data=customer_id, **kwargs)
        content = await resp.text()
    except ClientConnectorError:
        return {'customer_id': customer_id, 'prediction': ''}
    return ast.literal_eval(content)


async def make_requests(customer_ids: set, **kwargs) -> tuple:
    async with ClientSession() as session:
        tasks = []
        for customer_id in customer_ids:
            tasks.append(
                make_prediction_request(customer_id, session=session, **kwargs)
            )
        results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    from datetime import datetime
    start = datetime.utcnow()

    test_customer_ids = set(pd.read_csv('data/customers_1000.csv')['customer_id'])
    predictions = asyncio.run(make_requests(test_customer_ids))

    print(predictions)
    print('Total time:', str(datetime.utcnow() - start))
