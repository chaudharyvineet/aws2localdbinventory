import boto3
from datetime import datetime
import json
from typing import List, Dict, Any

def get_all_regions() -> List[str]:
    """Get list of all available AWS regions."""
    ec2_client = boto3.client('ec2')
    regions = [region['RegionName'] for region in ec2_client.describe_regions()['Regions']]
    return regions

def get_resource_inventory(region: str) -> List[Dict[Any, Any]]:
    """Get resource inventory for a specific region."""
    config_client = boto3.client('config', region_name=region)
    resources = []
    
    # Get list of supported resource types
    try:
        resource_types = config_client.get_discovered_resource_counts()['resourceCounts']
        resource_types = [item['resourceType'] for item in resource_types]
    except Exception as e:
        print(f"Error getting resource types: {str(e)}")
        resource_types = []
    
    for resource_type in resource_types:
        try:
            paginator = config_client.get_paginator('list_discovered_resources')
            
            # List resources of this type
            for page in paginator.paginate(resourceType=resource_type):
                for resource in page['resourceIdentifiers']:
                    try:
                        # Get detailed history for each resource
                        history = config_client.get_resource_config_history(
                            resourceType=resource['resourceType'],
                            resourceId=resource['resourceId']
                        )
                        
                        for config_item in history['configurationItems']:
                            # Extract creator/deleter information from CloudTrail events if available
                            creator_info = "N/A"
                            deleter_info = "N/A"
                            
                            if 'relatedEvents' in config_item:
                                for event in config_item['relatedEvents']:
                                    if 'CreateResource' in event:
                                        creator_info = config_item.get('configuration', {}).get('userIdentity', {}).get('arn', 'N/A')
                                    elif 'DeleteResource' in event:
                                        deleter_info = config_item.get('configuration', {}).get('userIdentity', {}).get('arn', 'N/A')
                            
                            resources.append({
                                'resourceType': config_item['resourceType'],
                                'resourceId': config_item['resourceId'],
                                'resourceName': config_item.get('resourceName', 'N/A'),
                                'region': region,
                                'accountId': config_item['accountId'],
                                'configurationItemStatus': config_item['configurationItemStatus'],
                                'resourceCreationTime': config_item.get('resourceCreationTime', 'N/A'),
                                'configurationItemCaptureTime': config_item['configurationItemCaptureTime'],
                                'awsRegion': config_item['awsRegion'],
                                'creator': creator_info,
                                'deleter': deleter_info,
                                'tags': config_item.get('tags', {}),
                                'relatedEvents': config_item.get('relatedEvents', [])
                            })
                    except config_client.exceptions.ResourceNotRecordedException:
                        continue
                    except Exception as e:
                        print(f"Error processing resource {resource['resourceId']}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error processing resource type {resource_type}: {str(e)}")
            continue
    
    return resources

def generate_html_report(resources: List[Dict[Any, Any]], output_file: str = 'resource_inventory.html'):
    """Generate HTML report from resource inventory."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AWS Resource Inventory Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f5f5f5; }
            .deleted { background-color: #ffe6e6; }
            .active { background-color: #e6ffe6; }
            .summary { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>AWS Resource Inventory Report</h1>
        <div class="summary">
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>Total Resources: """ + str(len(resources)) + """</p>
        </div>
        <table>
            <tr>
                <th>Resource Type</th>
                <th>Resource ID</th>
                <th>Resource Name</th>
                <th>Region</th>
                <th>Status</th>
                <th>Creation Time</th>
                <th>Last Modified</th>
                <th>Creator</th>
                <th>Deleter</th>
                <th>Tags</th>
            </tr>
    """
    
    for resource in resources:
        status_class = 'deleted' if resource['configurationItemStatus'] == 'ResourceDeleted' else 'active'
        html_content += f"""
            <tr class="{status_class}">
                <td>{resource['resourceType']}</td>
                <td>{resource['resourceId']}</td>
                <td>{resource['resourceName']}</td>
                <td>{resource['awsRegion']}</td>
                <td>{resource['configurationItemStatus']}</td>
                <td>{resource['resourceCreationTime']}</td>
                <td>{resource['configurationItemCaptureTime']}</td>
                <td>{resource['creator']}</td>
                <td>{resource['deleter']}</td>
                <td>{json.dumps(resource['tags'], indent=2)}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    """Main function to run the resource inventory collection and report generation."""
    # Initialize AWS credentials (assumes credentials are configured)
    try:
        all_resources = []
        regions = get_all_regions()
        
        print(f"Found {len(regions)} regions. Starting resource inventory...")
        
        for region in regions:
            print(f"Processing region: {region}")
            resources = get_resource_inventory(region)
            all_resources.extend(resources)
            print(f"Found {len(resources)} resources in {region}")
        
        print("Generating HTML report...")
        generate_html_report(all_resources)
        print("Report generated successfully as 'resource_inventory.html'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
