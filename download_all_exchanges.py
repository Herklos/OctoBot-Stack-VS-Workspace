#!/usr/bin/env python3
#  Drakkar-Software OctoBot-Tentacles
#  Copyright (c) Drakkar-Software, All rights reserved.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library.

import sys
import importlib.util
from pathlib import Path

# List of exchanges to download
EXCHANGES = ['bisq'] # 'polymarket', 

# List of tentacles folders to process (relative to workspace root)
# Examples: 'OctoBot/packages/tentacles', 'Private-Tentacles', 'Business-Private-Tentacles'
TENTACLES_FOLDERS = [
    'OctoBot/packages/tentacles',
]

def load_and_run_download_script(exchange_name: str, script_path: str):
    """Load and execute a download.py script for a given exchange."""
    print(f"\n{'='*60}")
    print(f"Downloading CCXT files for {exchange_name}...")
    print(f"{'='*60}\n")
    
    try:
        spec = importlib.util.spec_from_file_location(f"{exchange_name}_download", script_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load script from {script_path}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Execute the copy_files function directly since __name__ != '__main__' when imported
        if hasattr(module, 'copy_files'):
            module.copy_files()
            print(f"✓ Successfully processed {exchange_name}")
            return True
        else:
            print(f"Error: copy_files function not found in {script_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {exchange_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to download CCXT files for all exchanges."""
    workspace_root = Path(__file__).parent
    results = {}
    
    # Process each tentacles folder
    for tentacles_folder in TENTACLES_FOLDERS:
        base_path = workspace_root / tentacles_folder / 'Trading' / 'Exchange'
        
        print(f"\n{'#'*60}")
        print(f"Processing tentacles folder: {tentacles_folder}")
        print(f"{'#'*60}")
        
        if not base_path.exists():
            print(f"Warning: Tentacles folder not found at {base_path}")
            continue
        
        for exchange in EXCHANGES:
            script_path = base_path / exchange / 'script' / 'download.py'
            
            if not script_path.exists():
                print(f"Warning: Script not found for {exchange} at {script_path}")
                results[f"{tentacles_folder}/{exchange}"] = False
                continue
            
            results[f"{tentacles_folder}/{exchange}"] = load_and_run_download_script(exchange, str(script_path))
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{key}: {status}")
    
    print(f"{'='*60}\n")
    
    # Return exit code 0 if all succeeded, 1 if any failed
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
