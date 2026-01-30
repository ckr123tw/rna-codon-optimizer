import pandas as pd
import os

files = [
    "data/41467_2024_54059_MOESM3_ESM.xlsx",
    "data/NIHMS2101086-supplement-Supplementary_Table1.xlsx"
]

for f in files:
    print(f"\n{'='*20}\nFile: {f}\n{'='*20}")
    try:
        if os.path.exists(f):
            df = pd.read_excel(f, nrows=5)
            print(f"Columns for {os.path.basename(f)}:")
            seq_cols = [c for c in df.columns if any(x in c.lower() for x in ['seq', 'cds', 'utr', 'trans'])]
            print(f"  Potential sequence columns: {seq_cols}")
            print(f"  All columns: {df.columns.tolist()}")
        else:
            print("File not found.")
    except Exception as e:
        print(f"Error reading {f}: {e}")
