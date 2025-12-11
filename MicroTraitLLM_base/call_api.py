import requests

def call_api(url):
    # Function to call an API and return the data
    # Handles exceptions and returns None if the request fails
    try:
        # Send a GET request to the URL
        response = requests.get(url,timeout=60)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the data directly
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None
    
    except ValueError as e:
        print(f"Error parsing JSON from {url}: {e}")
        return None


# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.