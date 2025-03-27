#!/usr/bin/env python
"""
Script to fix common linting issues across the project.
"""

import os
import sys
import argparse
import re
from pathlib import Path


def fix_file(filepath):
    """Fix common linting issues in a file."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix missing final newline
    if not content.endswith('\n'):
        content += '\n'
    
    # Fix multiple consecutive blank lines (more than 2)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Ensure 2 blank lines after function/class definitions
    content = re.sub(r'(\)\s*:.*?\n\s*[^\s#].*?\n)(\s*[^\s#])', r'\1\n\n\2', content, flags=re.DOTALL)
    
    # Fix f-strings without placeholders
    content = re.sub(r'f"([^{]*?)"', r'"\1"', content)
    content = re.sub(r"f'([^{]*?)'", r"'\1'", content)
    
    # Write the fixed content back to the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix common linting issues across Python files.")
    parser.add_argument('--path', default='src', help='Path to search for Python files (default: src)')
    args = parser.parse_args()
    
    # Find all Python files
    python_files = []
    for root, _, files in os.walk(args.path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    # Fix each file
    for file in python_files:
        fix_file(file)
    
    print("All files processed!")


if __name__ == "__main__":
    main() 