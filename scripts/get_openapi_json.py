import requests
import json
import os

def fetch_openapi_json():
    url = "http://localhost:5002/openapi.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        openapi_data = response.json()
        
        # Ensure the docs-quarto directory exists
        os.makedirs("docs-quarto", exist_ok=True)
        
        # Save the OpenAPI JSON to the docs-quarto directory
        with open("docs-quarto/openapi.json", "w") as f:
            json.dump(openapi_data, f, indent=2)
        
        print("OpenAPI JSON successfully fetched and saved to docs-quarto/openapi.json")
    except requests.RequestException as e:
        print(f"Error fetching OpenAPI JSON: {e}")
    except json.JSONDecodeError:
        print("Error: Received invalid JSON from the server")
    except IOError as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    fetch_openapi_json()