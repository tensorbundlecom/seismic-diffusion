#!/usr/bin/env python
"""
Script to combine GSE phase pick files grouped by year.
Each year's files are merged and saved separately in data/phase_picks/years/
"""
import os
from pathlib import Path
from obspy import read_events
import pickle
from collections import defaultdict

def combine_gse_files_by_year(input_dir, output_dir):
    """
    Read all GSE .txt files from input_dir, group them by year,
    and combine each year's files into a separate output file.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing GSE files
    output_dir : str or Path
        Directory where year-based combined files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files in the directory
    gse_files = sorted(input_path.glob("*.txt"))
    
    if not gse_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    # Group files by year (first 4 characters of filename)
    files_by_year = defaultdict(list)
    for gse_file in gse_files:
        year = gse_file.name[:4]
        files_by_year[year].append(gse_file)
    
    print(f"Found {len(gse_files)} GSE files to combine")
    print(f"Years found: {sorted(files_by_year.keys())}")
    
    # Process each year separately
    for year in sorted(files_by_year.keys()):
        year_files = files_by_year[year]
        output_file = output_path / f"{year}.pkl"
        
        print(f"\n{'='*60}")
        print(f"Processing year {year}: {len(year_files)} files")
        print(f"{'='*60}")
        
        combined_catalog = None
        
        for i, gse_file in enumerate(year_files, 1):
            print(f"  [{i}/{len(year_files)}] Processing: {gse_file.name}")
            
            # Read the GSE file using ObsPy
            try:
                catalog = read_events(str(gse_file), format="GSE2")
                print(f"      → Read {len(catalog)} events")
            except Exception as e:
                print(f"      ✗ Error reading {gse_file.name}: {e}")
                continue
            
            if combined_catalog is None:
                combined_catalog = catalog
            else:
                # Extend the combined catalog with events from current file
                combined_catalog.extend(catalog)
        
        if combined_catalog is None:
            print(f"  ✗ No events were successfully read for year {year}")
            continue
        
        # Apply magnitude filter
        print(f"\n  Total events before filtering: {len(combined_catalog)}")
        combined_catalog = combined_catalog.filter("magnitude >= 1.0")
        print(f"  Total events after filtering (mag>1.0): {len(combined_catalog)}")
        
        # Write the combined catalog to output file
        output_file = output_path / f"{year}.pkl"
        print(f"  Writing to: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump(combined_catalog, f)
        
        print(f"  ✓ Successfully created: {output_file}")

if __name__ == "__main__":
    # Define input and output paths
    input_directory = "data/phase_picks"
    output_directory = "data/phase_picks/years"
    
    # Combine files grouped by year
    combine_gse_files_by_year(input_directory, output_directory)
