import pandas as pd
import os

files = [
    "data/41467_2024_54059_MOESM3_ESM.xlsx",
    "data/NIHMS2101086-supplement-Supplementary_Table1.xlsx"
]

# Keywords that indicate tissue (not cell line)
tissue_keywords = ['tissue', 'muscle', 'liver', 'brain', 'kidney', 'heart', 'lung', 'prostate', 'neurons']
cell_line_keywords = ['hek', 'hela', 'hepg', 'mcf', 'a549', 'k562', 'u2os', 'jurkat', 'thp', 'pc3', 'calu']

for f in files:
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(f)}")
    print('='*60)
    
    if not os.path.exists(f):
        print("File not found!")
        continue
        
    df = pd.read_excel(f, nrows=5)
    
    # Find TE columns
    te_cols = [c for c in df.columns if c.startswith('TE_') or 'te' in c.lower()]
    
    tissue_cols = []
    cell_line_cols = []
    unknown_cols = []
    
    for col in te_cols:
        col_lower = col.lower()
        is_tissue = any(kw in col_lower for kw in tissue_keywords)
        is_cell = any(kw in col_lower for kw in cell_line_keywords)
        
        if is_tissue:
            tissue_cols.append(col)
        elif is_cell:
            cell_line_cols.append(col)
        else:
            unknown_cols.append(col)
    
    print(f"\nTOTAL TE COLUMNS: {len(te_cols)}")
    print(f"\nTISSUE-SPECIFIC COLUMNS ({len(tissue_cols)}):")
    for c in tissue_cols:
        print(f"  - {c}")
    
    print(f"\nCELL LINE COLUMNS ({len(cell_line_cols)}):")
    for c in cell_line_cols[:10]:  # Show first 10
        print(f"  - {c}")
    if len(cell_line_cols) > 10:
        print(f"  ... and {len(cell_line_cols)-10} more")
    
    print(f"\nUNCLASSIFIED ({len(unknown_cols)}):")
    for c in unknown_cols:
        print(f"  - {c}")
