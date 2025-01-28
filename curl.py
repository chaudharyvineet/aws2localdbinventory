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





import boto3
import json
import random
import os
import time
from botocore.exceptions import ClientError
from datetime import datetime, timezone

AMI_ID = ''        
SECURITY_GROUP_ID = '' 
IAM_ROLE = ''         
VPC_ID = ''       
KEY_NAME = '-arm'
STORAGE_SIZE = 150

# DynamoDB table for tracking provision requests
REQUESTS_TABLE = 'instance-provision-requests'

ec2_client = boto3.client('ec2')
dynamodb = boto3.resource('dynamodb')
requests_table = dynamodb.Table(REQUESTS_TABLE)

INSTANCE_TYPE_FALLBACKS = {
    't4g.2xlarge': ['t4g.xlarge', 't4g.large', 't4g.medium'],
    't3.2xlarge': ['t3.xlarge', 't3.large', 't3.medium']
}

def parse_body(event):
    if 'body' in event:
        return json.loads(event['body'])
    return {}

def get_availability_zones():
    response = ec2_client.describe_availability_zones(
        Filters=[{'Name': 'region-name', 'Values': [ec2_client.meta.region_name]}]
    )
    return [az['ZoneName'] for az in response['AvailabilityZones']]

def get_subnet_for_az(az):
    subnets = ec2_client.describe_subnets(
        Filters=[
            {'Name': 'vpc-id', 'Values': [VPC_ID]},
            {'Name': 'availability-zone', 'Values': [az]}
        ]
    )['Subnets']
    return random.choice(subnets)['SubnetId'] if subnets else None

def get_ami_and_instance_type(body):
    arch = body.get('arch')
    if arch == "graviton":
        instance_type = body.get('instance_type', 't4g.2xlarge')
        ami_id = "ami-xxx"
    else:
        instance_type = body.get('instance_type', 't3.2xlarge')
        ami_id = "ami-xxxx"
    return ami_id, instance_type

def create_provision_request(body, is_spot):
    request_id = f"req-{int(time.time())}-{random.randint(1000, 9999)}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    requests_table.put_item(Item={
        'request_id': request_id,
        'status': 'pending',
        'creation_time': timestamp,
        'last_updated': timestamp,
        'body': body,
        'is_spot': is_spot,
        'attempt_count': 0,
        'current_instance_type': body.get('instance_type'),
        'errors': [],
        'instance_id': None
    })
    
    return request_id

def try_launch_instance(request_id):
    # Get request details
    response = requests_table.get_item(Key={'request_id': request_id})
    if 'Item' not in response:
        return None
    
    request = response['Item']
    body = request['body']
    is_spot = request['is_spot']
    
    # Get configuration
    ami_id, initial_instance_type = get_ami_and_instance_type(body)
    availability_zones = get_availability_zones()
    instance_types = [initial_instance_type] + INSTANCE_TYPE_FALLBACKS.get(initial_instance_type, [])
    
    # Start from the last attempted configuration
    current_type_index = instance_types.index(request['current_instance_type'])
    instance_types = instance_types[current_type_index:]
    
    for instance_type in instance_types:
        for az in availability_zones:
            subnet_id = get_subnet_for_az(az)
            if not subnet_id:
                continue
                
            try:
                instance_params = {
                    'ImageId': ami_id,
                    'InstanceType': instance_type,
                    'KeyName': KEY_NAME,
                    'SecurityGroupIds': [SECURITY_GROUP_ID],
                    'SubnetId': subnet_id,
                    'MaxCount': 1,
                    'MinCount': 1,
                    'IamInstanceProfile': {'Name': IAM_ROLE},
                    'Placement': {'AvailabilityZone': az}
                }
                
                if is_spot:
                    instance_params['InstanceMarketOptions'] = {
                        'MarketType': 'spot',
                        'SpotOptions': {
                            'SpotInstanceType': 'one-time',
                            'InstanceInterruptionBehavior': 'terminate'
                        }
                    }
                
                response = ec2_client.run_instances(**instance_params)
                instance = response['Instances'][0]
                
                # Update request status
                requests_table.update_item(
                    Key={'request_id': request_id},
                    UpdateExpression='SET #status = :status, instance_id = :instance_id, last_updated = :timestamp',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': 'launched',
                        ':instance_id': instance['InstanceId'],
                        ':timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                return instance['InstanceId']
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['InsufficientInstanceCapacity', 'InsufficientHostCapacity']:
                    # Update attempt count and errors
                    requests_table.update_item(
                        Key={'request_id': request_id},
                        UpdateExpression='SET attempt_count = attempt_count + :inc, errors = list_append(errors, :error), current_instance_type = :type, last_updated = :timestamp',
                        ExpressionAttributeValues={
                            ':inc': 1,
                            ':error': [f"{instance_type} in {az}: {error_code}"],
                            ':type': instance_type,
                            ':timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    continue
                else:
                    raise e
    
    # If all attempts failed, update status
    requests_table.update_item(
        Key={'request_id': request_id},
        UpdateExpression='SET #status = :status, last_updated = :timestamp',
        ExpressionAttributeNames={'#status': 'status'},
        ExpressionAttributeValues={
            ':status': 'failed',
            ':timestamp': datetime.now(timezone.utc).isoformat()
        }
    )
    return None

def lambda_handler(event, context):
    token = event['headers'].get('Authorization')
    secret = os.environ.get('header')
    
    if not (token and token == secret):
        return {
            'statusCode': 401,
            'body': json.dumps('401 Unauthorized')
        }
    
    try:
        body = parse_body(event)
        
        if event['resource'] == '/provision-machine':
            print('create ec2 api was called...')
            
            is_spot = body.get('spot', False)
            request_id = create_provision_request(body, is_spot)
            
            # Start the provisioning attempt
            try_launch_instance(request_id)
            
            return {
                'statusCode': 202,
                'body': json.dumps({
                    'request_id': request_id,
                    'status': 'pending'
                })
            }
            
        elif event['resource'] == '/checkstatus':
            print('check status api was called...')
            
            request_id = body.get('request_id')
            if not request_id:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Request ID is required'})
                }
            
            # Get request status from DynamoDB
            response = requests_table.get_item(Key={'request_id': request_id})
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'body': json.dumps({'error': 'Request not found'})
                }
            
            request = response['Item']
            status_response = {
                'request_id': request_id,
                'status': request['status'],
                'attempt_count': request['attempt_count'],
                'current_instance_type': request['current_instance_type']
            }
            
            # If instance exists, get its details
            if request['instance_id']:
                try:
                    instance_details = ec2_client.describe_instances(InstanceIds=[request['instance_id']])
                    instance = instance_details['Reservations'][0]['Instances'][0]
                    status_response.update({
                        'instance_id': request['instance_id'],
                        'instance_state': instance['State']['Name'],
                        'public_ip': instance.get('PublicIpAddress', 'No public IP'),
                        'availability_zone': instance['Placement']['AvailabilityZone']
                    })
                except ClientError:
                    status_response['instance_state'] = 'not_found'
            
            # If status is pending, try another launch attempt
            if request['status'] == 'pending':
                try_launch_instance(request_id)
            
            return {
                'statusCode': 200,
                'body': json.dumps(status_response)
            }
            
        elif event['resource'] == '/delete-machine':
            print('delete ec2 api was called...')
            
            request_id = body.get('request_id')
            if not request_id:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Request ID is required'})
                }
            
            # Get request details
            response = requests_table.get_item(Key={'request_id': request_id})
            if 'Item' not in response or not response['Item'].get('instance_id'):
                return {
                    'statusCode': 404,
                    'body': json.dumps({'error': 'Instance not found'})
                }
            
            instance_id = response['Item']['instance_id']
            ec2_client.terminate_instances(InstanceIds=[instance_id])
            
            # Update request status
            requests_table.update_item(
                Key={'request_id': request_id},
                UpdateExpression='SET #status = :status, last_updated = :timestamp',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'terminating',
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            return {
                'statusCode': 202,
                'body': json.dumps({
                    'request_id': request_id,
                    'instance_id': instance_id,
                    'status': 'terminating'
                })
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
