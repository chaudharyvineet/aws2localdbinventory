import json
import boto3
import base64

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # Extract the bucket name and file key from the path parameters
        bucket_name = event['pathParameters']['bucketname']
        file_key = event['pathParameters']['filename']
        
        # Extract action from query parameters
        action = event['queryStringParameters']['action']
        
        if action == 'initiate':
            return initiate_multipart_upload(bucket_name, file_key)
        elif action == 'uploadPart':
            return upload_part(event, bucket_name, file_key)
        elif action == 'complete':
            return complete_multipart_upload(event, bucket_name, file_key)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid action'})
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': str(e)})
        }

def initiate_multipart_upload(bucket_name, file_key):
    # Initiate multipart upload
    response = s3_client.create_multipart_upload(Bucket=bucket_name, Key=file_key)
    upload_id = response['UploadId']
    
    return {
        'statusCode': 200,
        'body': json.dumps({'uploadId': upload_id})
    }

def upload_part(event, bucket_name, file_key):
    # Extract parameters from the request
    body = json.loads(event['body'])
    upload_id = body['upload_id']
    part_number = body['part_number']
    part_data = base64.b64decode(body['part_data'])
    
    # Upload part
    response = s3_client.upload_part(
        Bucket=bucket_name,
        Key=file_key,
        PartNumber=part_number,
        UploadId=upload_id,
        Body=part_data
    )
    
    # Return ETag to the client
    return {
        'statusCode': 200,
        'body': json.dumps({'ETag': response['ETag']})
    }

def complete_multipart_upload(event, bucket_name, file_key):
    # Extract parameters from the request
    body = json.loads(event['body'])
    upload_id = body['upload_id']
    parts = body['parts']
    
    # Format parts for the complete call
    multipart_upload = {
        'Parts': [{'ETag': part['ETag'], 'PartNumber': part['PartNumber']} for part in parts]
    }
    
    # Complete multipart upload
    response = s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=file_key,
        UploadId=upload_id,
        MultipartUpload=multipart_upload
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }





################
curl -X POST https://api.com/your-bucketname/example.txt?action=uploadPart \
    -H "Content-Type: application/json" \
    -d '{"upload_id": "your-upload-id", "part_number": 1, "part_data": "base64-encoded-data"}'
