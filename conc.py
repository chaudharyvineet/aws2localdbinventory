import requests
import time
from concurrent.futures import ThreadPoolExecutor

# API Gateway URL and API key
API_URL = ""
API_KEY = ""

# Number of requests to send per second
REQUESTS_PER_SECOND = 200
# Total number of requests you want to send
TOTAL_REQUESTS = 1000

# Function to make a GET request with x-api-key header
def make_request():
    headers = {
        'x-api-key': API_KEY,
    }
    try:
        response = requests.get(API_URL, headers=headers)
        print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"Request failed: {e}")

# Function to send multiple requests concurrently
def send_requests_concurrently(total_requests, requests_per_second):
    # Calculate how many threads to run per second (request rate)
    with ThreadPoolExecutor(max_workers=requests_per_second) as executor:
        futures = []
        for _ in range(total_requests):
            futures.append(executor.submit(make_request))
            # Sleep between bursts of requests to keep the rate steady
            time.sleep(1 / requests_per_second)

        # Wait for all requests to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    # Send requests
    send_requests_concurrently(TOTAL_REQUESTS, REQUESTS_PER_SECOND)
