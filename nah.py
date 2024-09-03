import boto3
import json
import time

# Hardcoded values
AMI_ID = 'ami-xxxxxxxx'            # Replace with your AMI ID
SECURITY_GROUP_ID = 'sg-xxxxxxxx'  # Replace with your Security Group ID
IAM_ROLE = 'your-iam-role'         # Replace with your IAM role name
SUBNET_ID = 'subnet-xxxxxxxx'      # Optional: Replace with your Subnet ID if needed

ec2_client = boto3.client('ec2')

def create_spot_instance(event, context):
    try:
        # Extract the instance name and instance type from the event
        instance_name = event['name']
        instance_type = event.get('instance_type', 't2.micro')  # Default to 't2.micro' if not provided

        # Create the Spot Instance request
        spot_response = ec2_client.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification={
                'ImageId': AMI_ID,
                'InstanceType': instance_type,
                'SecurityGroupIds': [SECURITY_GROUP_ID],
                'IamInstanceProfile': {
                    'Name': IAM_ROLE
                },
                'SubnetId': SUBNET_ID,
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {
                                'Key': 'Name',
                                'Value': instance_name
                            }
                        ]
                    }
                ]
            }
        )

        spot_request_id = spot_response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

        # Wait until the Spot Instance request is fulfilled and the instance is running
        instance_id = None
        for _ in range(10):  # Poll for a maximum of ~50 seconds
            spot_result = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            if 'InstanceId' in spot_result['SpotInstanceRequests'][0]:
                instance_id = spot_result['SpotInstanceRequests'][0]['InstanceId']
                break
            time.sleep(5)

        if not instance_id:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Instance could not be launched'})
            }

        # Wait until the instance status is 'running'
        ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])

        # Get the instance details
        instance_details = ec2_client.describe_instances(InstanceIds=[instance_id])
        public_ip = instance_details['Reservations'][0]['Instances'][0].get('PublicIpAddress', 'No public IP')

        # Return the details
        return {
            'statusCode': 200,
            'body': json.dumps({
                'SpotInstanceRequestId': spot_request_id,
                'InstanceId': instance_id,
                'PublicIpAddress': public_ip
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def delete_ec2_instance(event, context):
    try:
        instance_name = event['name']

        # Describe instances using the provided name as a filter
        response = ec2_client.describe_instances(
            Filters=[
                {
                    'Name': 'tag:Name',
                    'Values': [instance_name]
                }
            ]
        )

        instance_ids = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])

        if not instance_ids:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Instance not found'})
            }

        # Terminate the instance(s)
        ec2_client.terminate_instances(InstanceIds=instance_ids)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Instance(s) terminated', 'InstanceIds': instance_ids})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def lambda_handler(event, context):
    # Determine if we're creating or deleting an instance
    if event['resource'] == '/createec2':
        return create_spot_instance(event, context)
    elif event['resource'] == '/deleteec2':
        return delete_ec2_instance(event, context)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Unsupported operation'})
        }
