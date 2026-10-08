import re
import pandas as pd
import numpy as np

def parse_arff(arff_path):
    attributes = []
    data_rows = []
    data_started = False
    
    with open(arff_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@attribute'):
                parts = line.split()
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
            elif line.lower().startswith('@data'):
                data_started = True
            elif data_started:
                # Handle comma separated values, respecting potential quotes
                # Using csv parser logic or regex
                items = [item.strip().strip("'\"") for item in line.split(',')]
                if len(items) == len(attributes):
                    data_rows.append(items)
                else:
                    # fallback csv regex
                    parsed = re.findall(r"(?:[^\s,']|'(?:\\.|[^'])*')+", line)
                    cleaned = [p.strip().strip("'\"") for p in parsed]
                    if len(cleaned) == len(attributes):
                        data_rows.append(cleaned)
                        
    df = pd.DataFrame(data_rows, columns=attributes)
    return df

if __name__ == '__main__':
    df = parse_arff('data/autism_adult/Autism-Adult-Data.arff')
    print("Parsed Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Target Distribution:")
    print(df['Class/ASD'].value_counts())
    
    # Save primary dataset
    df.to_csv('data/autism_screening_data.csv', index=False)
    print("Saved to data/autism_screening_data.csv successfully!")
