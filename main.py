import os
import argparse
import re
import sys
import csv
import boto3

parser = argparse.ArgumentParser()
# parser.add_argument('-r', help='run in single account mode or multiple', default="single")
parser.add_argument('-o', '--output_file', help='output file', default='db/aws_resources.csv')
parser.add_argument('-c', '--credentials_file', help='read credentials from here', dest='credentials_file')
# parser.add_argument('-b', '--blacklist', help='remove these extensions', dest='blacklist', nargs='+')
# parser.add_argument('-f', '--filters', help='additional filters, read docs', dest='filters', nargs='+')
args = parser.parse_args()



def set_aws_credentials():
    if args.credentials_file:
        try:
            with open(args.credentials_file, 'r') as file:
                reader = csv.reader(file)
                account_id, aws_access_key, aws_secret_key = next(reader)
                os.environ["AWS_ACCOUNT_ID"] = account_id
                os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
            print("AWS credentials set successfully from file!")
        except Exception as e:
            print(f"Error reading credentials from file: {e}")
    else:
        existing_aws_account_id = os.environ.get('AWS_ACCOUNT_ID')
        existing_aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        existing_aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if existing_aws_account_id and existing_aws_access_key and existing_aws_secret_key:
            change_credentials = input("AWS credentials are already set. Do you want to change them? (y/n): ").strip().lower()
            if change_credentials != 'y':
                return
        account_id = input("Enter AWS Account ID: ")
        aws_access_key = input("Enter AWS Access Key: ")
        aws_secret_key = input("Enter AWS Secret Key: ")
        os.environ["AWS_ACCOUNT_ID"] = account_id
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        print("AWS credentials set successfully!")

# def main():
#     set_aws_credentials()
def list_resources(region=None):
    client = boto3.client('resourcegroupstaggingapi', region_name=region) if region else boto3.client(
        'resourcegroupstaggingapi')

    try:
        resources = []
        paginator = client.get_paginator('get_resources')
        response_iterator = paginator.paginate(
            PaginationConfig={
                'PageSize': 50,  # You can adjust the number of resources per page as needed
            },
            ResourceTypeFilters=[],
            ResourcesPerPage=123,
            IncludeComplianceDetails=True
        )

        for response in response_iterator:
            print("Response:", response)  # Print the entire response
            resources.extend(response.get('ResourceTagMappingList', []))

        if not resources:
            print("No resources found.")
            return



        with open(args.output_file, 'w', newline='') as csv_file:
            fieldnames = ['ResourceARN', 'ResourceType', 'Tags']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()

            for resource in resources:
                print("Resource:", resource)  # Print each resource
                writer.writerow({
                    'ResourceARN': resource['ResourceARN'],
                    'ResourceType': resource.get('ResourceType', 'N/A'),  # Handle missing ResourceType
                    'Tags': resource.get('Tags', 'No tags'),
                })

        print(f"Total resources found: {len(resources)}")
        print(f"Resource information saved to {args.output_file}")

    except Exception as e:
        print(f"Error listing resources: {e}")


if __name__ == "__main__":
    set_aws_credentials()
    region_input = input("Enter a specific region (leave blank for all regions): ").strip()

    if region_input:
        list_resources(region=region_input)
    else:
        # Loop through all regions
        for region in boto3.Session().get_available_regions('resourcegroupstaggingapi'):
            print(f"\nResources in region: {region}")
            list_resources(region=region)
