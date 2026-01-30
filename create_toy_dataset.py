import pandas as pd
import numpy as np
import os

def create_toy():
    zheng_file = "data/41467_2024_54059_MOESM3_ESM.xlsx"
    atlas_file = "data/NIHMS2101086-supplement-Supplementary_Table1.xlsx"
    output_file = "data/toy_dataset.csv"
    
    dfs = []
    
    # Process Zheng et al. (File 1)
    if os.path.exists(zheng_file):
        print(f"Loading {zheng_file}...")
        df1 = pd.read_excel(zheng_file, nrows=200) # Sample 200
        # Columns: 'mRNA Sequence', 'half-life', 'k_fit' (assume TE proxy)
        # Check actual columns from inspection
        # "mRNA Sequence"
        # "half-life"
        # "k_fit" ?? Or "mean_te"? Inspection showed "k_fit". 
        # Let's verify with standard column names or permissive lookup
        
        rename_map = {}
        if "mRNA Sequence" in df1.columns:
            rename_map["mRNA Sequence"] = "sequence"
        
        # Look for TE-like column
        for c in df1.columns:
            if "k_fit" in c or "TE" in c:
                rename_map[c] = "TE"
                break
        
        if "half-life" in df1.columns:
            rename_map["half-life"] = "HalfLife"
            
        if "sequence" in rename_map.values():
            df1 = df1.rename(columns=rename_map)
            # Filter valid
            cols = ["sequence"]
            if "TE" in df1.columns: cols.append("TE")
            if "HalfLife" in df1.columns: cols.append("HalfLife")
            
            df1 = df1[cols].copy()
            df1["cell_line"] = "HEK293" # Default for Zheng
            # Fill missing
            if "TE" not in df1.columns: df1["TE"] = np.random.rand(len(df1)) * 5
            if "HalfLife" not in df1.columns: df1["HalfLife"] = np.random.rand(len(df1)) * 10
            
            dfs.append(df1)
            print(f"Added {len(df1)} samples from Zheng et al.")
            
    # Process Atlas (File 2)
    if os.path.exists(atlas_file):
        print(f"Loading {atlas_file}...")
        df2 = pd.read_excel(atlas_file, nrows=200)
        # Columns: "tx_sequence", "TE_HeLa", "TE_HepG2", ...
        
        if "tx_sequence" in df2.columns:
            df2 = df2.rename(columns={"tx_sequence": "sequence"})
            
            # Melt cell lines
            cell_cols = [c for c in df2.columns if c.startswith("TE_")]
            if cell_cols:
                # Take cell lines AND tissues for toy data
                target_cols = [
                    # Cell lines
                    "TE_HeLa", "TE_HepG2", "TE_HEK293",
                    # Tissues
                    "TE_muscle_tissue", "TE_skeletal_muscle", 
                    "TE_normal_brain_tissue", "TE_neurons",
                    "TE_Kidney_normal_tissue"
                ]
                target_cols = [c for c in target_cols if c in cell_cols]
                
                if target_cols:
                    df_melted = df2.melt(
                        id_vars=["sequence"], 
                        value_vars=target_cols,
                        var_name="context_col",
                        value_name="TE"
                    )
                    # Extract context name (remove TE_ prefix)
                    df_melted["cell_line"] = df_melted["context_col"].str.replace("TE_", "")
                    
                    # Atlas doesn't have HalfLife? Impute.
                    df_melted["HalfLife"] = np.random.rand(len(df_melted)) * 10
                    
                    dfs.append(df_melted[["sequence", "TE", "HalfLife", "cell_line"]])
                    print(f"Added {len(df_melted)} samples from Atlas (including tissues).")
                
    if not dfs:
        print("No data found! Creating random mock.")
        df = pd.DataFrame({
            "sequence": ["AUG" + "C"*30]*10,
            "TE": np.random.rand(10),
            "HalfLife": np.random.rand(10),
            "cell_line": ["HEK293"]*10
        })
    else:
        df = pd.concat(dfs, ignore_index=True)
        
    # Clean sequences
    df = df.dropna(subset=["sequence"])
    # Ensure they are strings
    df["sequence"] = df["sequence"].astype(str)
    # Filter short sequences
    df = df[df["sequence"].str.len() > 10]
    
    print(f"Total samples: {len(df)}")
    print(df.head())
    
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    create_toy()
