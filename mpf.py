--code--
import json
import boto3
import cgi
from io import BytesIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # Extract the bucket name from the path parameters
        bucket_name = "bucketchy"
        # event['pathParameters']['bucket']

        # Extract the Content-Type header
        content_type_header = event['headers'].get('Content-Type', event['headers'].get('content-type'))
        if not content_type_header:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Content-Type header missing'})
            }

        # Extract the boundary string from the Content-Type header
        boundary = content_type_header.split("boundary=")[1]

        # Extract the body of the request
        body = event['body']

        # Parse the multipart form data using the boundary string
        environ = {'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': f'multipart/form-data; boundary={boundary}'}
        fp = BytesIO(body.encode())
        form = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)

        # Check if a file part exists in the form data
        if 'file' not in form:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'No file part in the request'})
            }

        file_item = form['file']
        if not file_item.file:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'File not found in the request'})
            }

        file_name = file_item.filename
        file_content = file_item.file.read()

        # Perform the upload to S3
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': f'File {file_name} uploaded successfully to bucket {bucket_name}'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': str(e)})
        }

----------------
model schema 

-----
{
  "type": "object",
  "properties": {
    "file": {
      "type": "string",
      "format": "binary"
    }
  },
  "required": ["file"]
}

----------------


{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:role"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::name/*"
        }
    ]
}


----------
curl -X POST api \
     -H "Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryABC123" \
     --data-binary @abcd.txt

------ abcd.txt

------WebKitFormBoundaryABC123
Content-Disposition: form-data; name="file"; filename="abcd.txt"
Content-Type: text/plain

This is the content of the file.
It can be multiple lines.
End of file content.
------WebKitFormBoundaryABC123--


import json
import boto3
import cgi
from io import BytesIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # Extract the bucket name from the path parameters
        bucket_name = "bucketchy"
        # event['pathParameters']['bucket']

        # Extract the Content-Type header
        content_type_header = event['headers'].get('Content-Type', event['headers'].get('content-type'))
        if not content_type_header:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Content-Type header missing'})
            }

        # Extract the boundary string from the Content-Type header
        boundary = content_type_header.split("boundary=")[1]

        # Extract the body of the request
        body = event['body']

        # Parse the multipart form data using the boundary string
        environ = {'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': f'multipart/form-data; boundary={boundary}'}
        fp = BytesIO(body.encode())
        form = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)

        # Check if a file part exists in the form data
        if 'file' not in form:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'No file part in the request'})
            }

        file_item = form['file']
        if not file_item.file:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'File not found in the request'})
            }

        file_name = file_item.filename
        file_content = file_item.file.read()

        # Perform the upload to S3
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': f'File {file_name} uploaded successfully to bucket {bucket_name}'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': str(e)})
        }

