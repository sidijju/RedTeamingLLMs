import json
import csv
import os
import sys
from datetime import datetime

# Function to convert JSON to CSV
def json_to_csv(json_path):
    # Extract the model name and timestamp from the JSON filename
    base_name = os.path.basename(json_path)
    name_parts = base_name.replace('.json', '').split('_')
    model_name = name_parts[0]
    
    # Create a timestamp for the output CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"result_{timestamp}_{model_name}.csv"
    
    # Load the JSON data
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Open the CSV file for writing
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header
        writer.writerow(["Instruction", "Harm Target", "Actor Name", "Query", "Model Response", "Score", "Reason"])
        
        # Iterate over each instruction in the data
        for entry in data['data']:
            instruction = entry.get('instruction', '')
            harm_target = entry.get('harm_target', '')
            
            # Process each attempt in the entry
            for attempt in entry.get('attempts', []):
                actor_name = attempt.get('actor', {}).get('actor', {}).get('actor_name', '')
                
                # Initialize model_response, score, and reason
                model_response, score, reason = "", "", ""
                
                # Process each dialog history item
                for dialog in attempt.get('dialog_hist', []):
                    if dialog['role'] == 'assistant':
                        model_response = dialog.get('content', '')
                        score = dialog.get('score', '')
                        reason = dialog.get('reason', '')
                    else:
                        query = dialog.get('content', '')
                        
                        # Write the row with instruction, harm target, actor, query, response, score, and reason
                        writer.writerow([instruction, harm_target, actor_name, query, model_response, score, reason])
    
    print(f"Data has been successfully exported to {csv_filename}")

# Main function to handle input
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python json_to_csv.py <path_to_json_file>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist.")
        sys.exit(1)
    
    json_to_csv(json_path)
