import ast
import os
from pathlib import Path

def find_imports(directory='.'):
    imports = set()
    
    for py_file in Path(directory).rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except Exception as e:
            pass
    
    return sorted(imports)

# Get all imports
all_imports = find_imports()

# Filter out standard library
import sys
stdlib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()

third_party = [imp for imp in all_imports if imp not in stdlib and not imp.startswith('_')]

print("Third-party packages used in your project:\n")
for imp in third_party:
    print(f"  {imp}")
