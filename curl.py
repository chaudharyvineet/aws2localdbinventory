<< keep the single line >>
curl -X POST <api.comhere>/prod/nothing -H "Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryABC123" --data-binary @abcd.txt


df
FROM public.ecr.aws/lambda/python:3.8

RUN pip install nltk

RUN python -c "import nltk; nltk.download('punkt')"

RUN yum install -y tree

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.handler"]


import os
import nltk
from nltk.tokenize import word_tokenize

# Download the punkt tokenizer to the /tmp directory
nltk.data.path.append('/tmp')
if not os.path.exists('/tmp/tokenizers/punkt'):
    nltk.download('punkt', download_dir='/tmp')

def handler(event, context):
    # Print the folder structure
    os.system("tree /tmp")

    text = event.get('text', 'Hello, world!')
    tokens = word_tokenize(text)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'tokens': tokens
        })
    }





shdhjahdbs



import requests
import json

# API Gateway URL
api_url = 'https://api.com/your-endpoint'  # Replace with your API Gateway URL

# Headers including Content-Type and API Key if required
headers = {
    'Content-Type': 'application/json',
    'x-api-key': 'your-api-key'  # Replace with your actual API key if required
}

# Payload to send in the body of the POST request
payload = {
    'name': 'my-instance',
    'instance_type': 't2.micro'
}

# Send POST request
response = requests.post(api_url, headers=headers, data=json.dumps(payload))

# Check response
if response.status_code == 200:
    print('Request was successful!')
    print('Response:', response.json())
else:
    print('Request failed with status code:', response.status_code)
    print('Response:', response.text)




