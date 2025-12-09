#!/usr/bin/env python3
"""
Script to convert all waveform files (mseed) to PNG images.
Maps data/waveforms/HH/abcd.mseed -> data/png/HH/abcd.png
"""

import os
from pathlib import Path
from obspy import read
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

def convert_waveform_to_png(mseed_path, png_path):
    """
    Convert a single waveform file to PNG.
    
    Args:
        mseed_path: Path to the input mseed file
        png_path: Path to the output PNG file
    """
    try:
        # Read the waveform
        st = read(str(mseed_path))
        
        # Create the plot
        fig = st.plot(handle=True, show=False)
        
        # Save as PNG
        fig.savefig(str(png_path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"Error converting {mseed_path}: {e}")
        return False

def main():
    # Define base directories
    waveforms_dir = Path("data/waveforms")
    png_dir = Path("data/png")
    
    # Find all mseed files
    mseed_files = list(waveforms_dir.glob("**/*.mseed"))
    
    if not mseed_files:
        print(f"No mseed files found in {waveforms_dir}")
        return
    
    print(f"Found {len(mseed_files)} waveform files to convert")
    
    # Process each file
    converted = 0
    failed = 0
    
    for mseed_path in tqdm(mseed_files):
        # Get the relative path from waveforms_dir
        relative_path = mseed_path.relative_to(waveforms_dir)
        
        # Create the corresponding PNG path
        png_path = png_dir / relative_path.with_suffix('.png')
        
        # Create parent directories if they don't exist
        png_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if PNG already exists (optional: skip or overwrite)
        if png_path.exists():
            # print(f"Skipping (already exists): {png_path}")
            continue
        
        # Convert the waveform
        # print(f"Converting: {mseed_path} -> {png_path}")
        if convert_waveform_to_png(mseed_path, png_path):
            converted += 1
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted}")
    print(f"Failed: {failed}")
    print(f"Skipped (already existed): {len(mseed_files) - converted - failed}")

if __name__ == "__main__":
    main()
