################ lam ##########
import json
import boto3
from datetime import datetime
import uuid

# Initialize DynamoDB and API Gateway management client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('ChatMessages')
apigatewaymanagementapi = boto3.client('apigatewaymanagementapi', endpoint_url='https://rvso8onl90.execute-api.us-east-1.amazonaws.com/dev/')

# Store connected clients in DynamoDB
connection_table = dynamodb.Table('Connections')

def handle_connect(event):
    connection_id = event['requestContext'].get('connectionId')
    print(f"Connection established: {connection_id}")

    # Add connection ID to DynamoDB
    connection_table.put_item(
        Item={
            'ConnectionId': connection_id
        }
    )

    return {
        'statusCode': 200,
        'body': json.dumps('Connected')
    }

def handle_disconnect(event):
    connection_id = event['requestContext'].get('connectionId')
    print(f"Connection closed: {connection_id}")

    # Remove connection ID from DynamoDB
    connection_table.delete_item(
        Key={
            'ConnectionId': connection_id
        }
    )

    return {
        'statusCode': 200,
        'body': json.dumps('Disconnected')
    }

def handle_new_message(event):
    connection_id = event['requestContext'].get('connectionId')
    body = json.loads(event.get('body', '{}'))
    
    if 'message' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps('Missing message field')
        }
    
    message = body['message']
    message_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Store the message in DynamoDB
    table.put_item(
        Item={
            'MessageId': message_id,
            'Content': message,
            'Timestamp': timestamp,
            'ConnectionId': connection_id
        }
    )

    # Broadcast the message to all connected clients
    broadcast_message(message)

    return {
        'statusCode': 200,
        'body': json.dumps('Message stored and broadcasted')
    }

def broadcast_message(message):
    # Retrieve all connected clients from DynamoDB
    response = connection_table.scan()
    connection_ids = [item['ConnectionId'] for item in response['Items']]

    for connection_id in connection_ids:
        try:
            apigatewaymanagementapi.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({'message': message})
            )
        except apigatewaymanagementapi.exceptions.GoneException:
            # Connection might be closed or invalid
            print(f"Connection {connection_id} is no longer valid. Removing...")
            connection_table.delete_item(
                Key={
                    'ConnectionId': connection_id
                }
            )

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    route_key = event['requestContext'].get('routeKey', None)

    if route_key == '$connect':
        return handle_connect(event)
    elif route_key == '$disconnect':
        return handle_disconnect(event)
    elif route_key == 'new-message':
        return handle_new_message(event)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid route')
        }


---------------------------------- auth 
import json


def lambda_handler(event, context):
    print(event)

    # Retrieve request parameters from the Lambda function input:
    headers = event['headers']
    queryStringParameters = event['queryStringParameters']
    stageVariables = event['stageVariables']
    requestContext = event['requestContext']

    # Parse the input for the parameter values
    tmp = event['methodArn'].split(':')
    apiGatewayArnTmp = tmp[5].split('/')
    awsAccountId = tmp[4]
    region = tmp[3]
    ApiId = apiGatewayArnTmp[0]
    stage = apiGatewayArnTmp[1]
    route = apiGatewayArnTmp[2]

    # Perform authorization to return the Allow policy for correct parameters
    # and the 'Unauthorized' error, otherwise.

    authResponse = {}
    condition = {}
    condition['IpAddress'] = {}

    token = event['headers'].get('Authorization')
    if token and "Bearer" in token:
        response = generateAllow('me', event['methodArn'])
        print('authorized')
        return json.loads(response)
    else:
        print('unauthorized')
        return 'unauthorized'

    # Help function to generate IAM policy


def generatePolicy(principalId, effect, resource):
    authResponse = {}
    authResponse['principalId'] = principalId
    if (effect and resource):
        policyDocument = {}
        policyDocument['Version'] = '2012-10-17'
        policyDocument['Statement'] = []
        statementOne = {}
        statementOne['Action'] = 'execute-api:Invoke'
        statementOne['Effect'] = effect
        statementOne['Resource'] = resource
        policyDocument['Statement'] = [statementOne]
        authResponse['policyDocument'] = policyDocument

    authResponse['context'] = {
        "stringKey": "stringval",
        "numberKey": 123,
        "booleanKey": True
    }

    authResponse_JSON = json.dumps(authResponse)

    return authResponse_JSON


def generateAllow(principalId, resource):
    return generatePolicy(principalId, 'Allow', resource)


def generateDeny(principalId, resource):
    return generatePolicy(principalId, 'Deny', resource)

