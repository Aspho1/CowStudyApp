#!/usr/bin/env python3
"""
Python code fixer for CowStudyApp - fixes common f-string syntax errors.
"""
import os
import re
import sys
# from pathlib import Path






def fix_fstring_errors(file_path):
    """Fix common f-string errors in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    original_content = content
    
    # Fix #1: Nested quotes - change inner quotes to single quotes
    # Example: f"Text {var[\'key\']}" → f"Text {var['key']}"
    content = re.sub(r'f(["\'])(.*?){\s*(\w+)\["(.*?)"\](.*?)}(.*?)\1', 
                     r'f\1\2{\3[\'\4\']\5}\6\1', content)
    
    # Fix #2: Comma string concatenation inside f-string
    # Example: f"Text {", "\.join(list)}" → f"Text {', '.join(list)}"
    content = re.sub(r'{\s*",\s*"\.', r'{", "\.', content)
    
    # Fix #3: Unescaped quotes inside f-string expressions
    # This is more complex and might need special handling
    
    # Check if anything changed
    if content != original_content:
        # Backup the original file
        backup_path = str(file_path) + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    return False

def scan_and_fix_directory(base_dir):
    """Scan all Python files in a directory tree and fix f-string errors."""
    fixed_files = []
    error_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    if fix_fstring_errors(file_path):
                        fixed_files.append(file_path)
                        print(f"Fixed: {file_path}")
                except Exception as e:
                    error_files.append((file_path, str(e)))
                    print(f"Error in {file_path}: {e}")
    
    return fixed_files, error_files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = '/var/www/personal_flask_website/CowStudyApp'
    
    print(f"Scanning directory: {base_dir}")
    fixed, errors = scan_and_fix_directory(base_dir)
    
    print(f"\nFixed {len(fixed)} files:")
    for path in fixed:
        print(f"  - {path}")
    
    print(f"\nEncountered errors in {len(errors)} files:")
    for path, error in errors:
        print(f"  - {path}: {error}")
