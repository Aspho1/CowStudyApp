# test_r_setup.py
from pathlib import Path
import subprocess
import os
import logging

logging.basicConfig(level=logging.DEBUG)

def test_r():
    r_path = Path(r"C:\Program Files\R\R-4.3.0")
    r_script = r_path / "bin" / "Rscript.exe"
    
    if not r_script.exists():
        print(f"Rscript not found at {r_script}")
        return
        
    tests = [
        ([str(r_script), "--version"], "R version check"),
        ([str(r_script), "-e", "1+1"], "Basic R computation"),
        ([str(r_script), "-e", "installed.packages()[,1]"], "List installed packages"),
    ]
    
    for cmd, desc in tests:
        print(f"\nTesting: {desc}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_r()