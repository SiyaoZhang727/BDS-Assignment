import requests
import csv
import re

def fetch_data():
    """Fetch data from the API and return the response in JSON format."""
    url = "https://www.ec.europa.eu/agrifood/api/fruitAndVegetable/prices?memberStateCodes=IT,ES,EL,EU&products=oranges,clementines,lemons,mandarins&beginDate=01/01/2004&endDate=25/11/2024"  
    
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()  # Return the response data as JSON
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print(response.text)
        return None


def process_data(data):
    """Process the data to calculate price_per_kg and remove unnecessary fields."""
    processed_data = []

    for record in data:
        try:
            # Ensure the price is converted to a float
            price = float(record["price"].replace("€", "").strip())  # Remove currency symbol if present

            # Extract numeric value from the unit field (e.g., 100 from "100KG")
            unit_value = int(re.search(r'\d+', record["unit"]).group())

            # Calculate price per unit
            price_per_kg = price / unit_value

            # Create a new record with the desired fields
            processed_record = {
                "year": int(record["beginDate"].split("/")[-1]),  # Extract year from beginDate
                "weekNumber": int(record["weekNumber"]),
                "location": record["memberStateName"],
                "price_per_kg": round(price_per_kg, 4),  # Round to 4 decimal places
                "product": record["product"],
                "variety": record["variety"],
                "description": record["description"],
            }

            processed_data.append(processed_record)

        except (ZeroDivisionError, AttributeError, ValueError) as e:
            # Handle missing or malformed fields gracefully
            print(f"Error processing record: {record}. Error: {e}")
            continue

    return processed_data


def save_to_csv(data, file_path):
    """Save the processed data to a CSV file."""
    fieldnames = ["year", "weekNumber", "location", "price_per_kg", "product", "variety", "description"]
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    # Fetch data from API
    print("Fetching data from the API...")
    raw_data = fetch_data()

    if raw_data:
        # Process the data
        print("Processing data...")
        processed_data = process_data(raw_data)

        # Save processed data to CSV
        output_file = "price_data.csv"
        print(f"Saving processed data to '{output_file}'...")
        save_to_csv(processed_data, output_file)

        print("Process completed successfully!")
    else:
        print("Data fetching failed. Process terminated.")
