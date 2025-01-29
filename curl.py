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














def try_launch_spot_instance(instance_type, ami_id):
    """
    Attempt to launch a spot instance across multiple AZs.
    Returns (instance_id, spot_request_id, az) on success, or (None, None, None) on failure.
    """
    availability_zones = get_availability_zones()
    
    for az in availability_zones:
        subnet_id = get_subnet_for_az(az)
        if not subnet_id:
            continue
            
        instance_params = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'KeyName': KEY_NAME,
            'SecurityGroupIds': [SECURITY_GROUP_ID],
            'SubnetId': subnet_id,
            'MaxCount': 1,
            'MinCount': 1,
            'IamInstanceProfile': {'Name': IAM_ROLE},
            'InstanceMarketOptions': {
                'MarketType': 'spot',
                'SpotOptions': {
                    'SpotInstanceType': 'one-time',
                    'InstanceInterruptionBehavior': 'terminate'
                }
            }
        }
        
        try:
            response = ec2_client.run_instances(**instance_params)
            instance_id = response['Instances'][0]['InstanceId']
            
            # Get spot request ID
            spot_requests = ec2_client.describe_spot_instance_requests(
                Filters=[{'Name': 'instance-id', 'Values': [instance_id]}]
            )
            if spot_requests['SpotInstanceRequests']:
                spot_request_id = spot_requests['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                return instance_id, spot_request_id, az
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['InsufficientInstanceCapacity', 'SpotMaxPriceTooLow']:
                print(f"Failed to launch in AZ {az}: {error_code}")
                continue
            raise  # Re-raise if it's a different error
            
    return None, None, None

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
            
            arch = body.get('arch')
            if arch == "graviton":
                instance_type = body.get('instance_type', 't4g.2xlarge')
                AMI_ID = "ami-xxx"
            else:
                instance_type = body.get('instance_type', 't3.2xlarge')
                AMI_ID = "ami-xxxx"

            is_spot = body.get('spot', True)
            
            if is_spot:
                # Try the requested instance type first
                instance_id, spot_request_id, az = try_launch_spot_instance(instance_type, AMI_ID)
                
                # If that fails, try fallback instance types
                if not instance_id:
                    fallback_types = INSTANCE_TYPE_FALLBACKS.get(instance_type, [])
                    for fallback_type in fallback_types:
                        instance_id, spot_request_id, az = try_launch_spot_instance(fallback_type, AMI_ID)
                        if instance_id:
                            break
                
                if not instance_id:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'error': 'No spot capacity available in any AZ or instance type'
                        })
                    }
                    
                return {
                    'statusCode': 202,
                    'body': json.dumps({
                        'RequestId': spot_request_id,
                        'InstanceId': instance_id,
                        'Status': 'pending',
                        'IsSpot': True,
                        'AvailabilityZone': az,
                        'InstanceType': instance_type
                    })
                }
            
            else:
                # On-demand instance - use first available AZ
                subnet_id = get_subnet_for_az(get_availability_zones()[0])
                response = ec2_client.run_instances(
                    ImageId=AMI_ID,
                    InstanceType=instance_type,
                    KeyName=KEY_NAME,
                    SecurityGroupIds=[SECURITY_GROUP_ID],
                    SubnetId=subnet_id,
                    MaxCount=1,
                    MinCount=1,
                    IamInstanceProfile={'Name': IAM_ROLE}
                )
                
                instance_id = response['Instances'][0]['InstanceId']
                return {
                    'statusCode': 202,
                    'body': json.dumps({
                        'RequestId': None,
                        'InstanceId': instance_id,
                        'Status': 'pending',
                        'IsSpot': False
                    })
                }

