import json
import pandas as pd

# Read JSONL file
data = []
with open('bf16_optimized_results_raw.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Write to Excel
df.to_excel('bf16_optimized_results_raw.xlsx', index=False)
print(f'Successfully converted {len(df)} rows to Excel')
print(f'Columns: {list(df.columns)}')
