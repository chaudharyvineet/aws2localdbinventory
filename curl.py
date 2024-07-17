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

