import json
import os

filepath = r'backend\data\papers.json'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Try to find the first valid JSON array
bracket_count = 0
in_string = False
escape = False
end_idx = -1

for i, char in enumerate(content):
    if escape:
        escape = False
        continue
    if char == '\\':
        escape = True
        continue
    if char == '"':
        in_string = not in_string
        continue
        
    if not in_string:
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i
                break

if end_idx != -1:
    valid_json_str = content[:end_idx+1]
    try:
        data = json.loads(valid_json_str)
        print(f"Successfully parsed {len(data)} papers.")
        
        # Write it back correctly
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("Fixed papers.json")
    except Exception as e:
        print(f"Error parsing sliced json: {e}")
else:
    print("Could not find a valid JSON array.")
