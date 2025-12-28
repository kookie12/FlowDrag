import os
import re
from pathlib import Path
from flask import Flask, render_template, send_from_directory
import json

app = Flask(__name__)

DATASET_PATH = "dataset/VFD_Bench_Dataset"

def custom_sort_key(folder_name):
    """
    Sorting key function:
    1. Split by _
    2. Sort by first part (alphabet)
    3. Sort by number in second part (if not found, find first number in all parts)
    4. Sort by number in last part
    """
    parts = folder_name.split('_')
    
    # First part (alphabet)
    first_part = parts[0] if parts else ''
    
    # Extract number from second part
    second_num = 0
    if len(parts) > 1:
        # Extract only numbers from second part
        numbers = re.findall(r'\d+', parts[1])
        if numbers:
            second_num = int(numbers[0])
        else:
            # If no number in second part, find first number in all parts
            for part in parts[1:]:
                numbers = re.findall(r'\d+', part)
                if numbers:
                    second_num = int(numbers[0])
                    break
    
    # Extract number from last part (0 if not found)
    last_num = 0
    if len(parts) > 1:
        # Extract only numbers from last part
        numbers = re.findall(r'\d+', parts[-1])
        if numbers:
            last_num = int(numbers[-1])  # Use last number
    
    return (first_part, second_num, last_num)

def get_all_folders():
    """Get all subdirectories in the dataset folder"""
    folders = []
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            folders.append(item)
    # Sort using custom sort function
    folders.sort(key=custom_sort_key)
    return folders

@app.route('/')
def index():
    folders = get_all_folders()
    return render_template('viewer.html', folders=folders, total=len(folders))

@app.route('/dataset/<path:filepath>')
def serve_image(filepath):
    return send_from_directory(DATASET_PATH, filepath)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')
