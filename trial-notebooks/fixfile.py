import json
import re

def debug_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_content = file.read()

    # Fix single quotes to double quotes
    content = re.sub(r"(?<!\\)'", '"', raw_content)
    
    # Remove trailing commas
    content = re.sub(r",\s*([\]}])", r"\1", content)

    # Split into lines for debugging
    lines = content.splitlines()
    repaired_lines = []
    for i, line in enumerate(lines, start=1):
        repaired_lines.append(line)
        try:
            # Join repaired lines and test if it forms valid JSON so far
            json.loads("\n".join(repaired_lines))
        except json.JSONDecodeError as e:
            print(f"Error in line {i}: {e}")
            print(f"Problematic line: {line}")
            # You can manually inspect or decide to skip this line
            continue

    # Join repaired lines into a valid JSON object
    try:
        fixed_content = "\n".join(repaired_lines)
        notebook = json.loads(fixed_content)  # Validate JSON
    except json.JSONDecodeError as e:
        print(f"Final repair failed: {e}")
        raise ValueError("Manual repair required. See errors above.")

    # Save the repaired notebook
    output_path = file_path.replace('.ipynb', '_repaired.ipynb')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(notebook, file, indent=4)

    print(f"Notebook successfully repaired and saved to {output_path}.")

# Path to corrupted file
input_path = '/home/ronit/Desktop/QROSS-project/trial-notebooks/QROSS_Adavanced_GCN_Training_model.ipynb'

debug_json(input_path)
