#!/usr/bin/env python3
"""
Static code validation script.
Catches common errors like undefined variables before runtime.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class ConfigUsageChecker(ast.NodeVisitor):
    """Check that functions using 'config' properly call get_config()."""
    
    def __init__(self):
        self.errors: List[Tuple[str, int, str]] = []
        self.current_function = None
        self.function_stack = []
    
    def visit_FunctionDef(self, node):
        self.function_stack.append({
            'name': node.name,
            'lineno': node.lineno,
            'has_get_config': False,
            'uses_config': False,
            'is_async': False,
        })
        self.generic_visit(node)
        func_info = self.function_stack.pop()
        
        if func_info['uses_config'] and not func_info['has_get_config']:
            self.errors.append((
                func_info['name'],
                func_info['lineno'],
                f"Function '{func_info['name']}' uses 'config' but doesn't call get_config()"
            ))
    
    def visit_AsyncFunctionDef(self, node):
        self.function_stack.append({
            'name': node.name,
            'lineno': node.lineno,
            'has_get_config': False,
            'uses_config': False,
            'is_async': True,
        })
        self.generic_visit(node)
        func_info = self.function_stack.pop()
        
        if func_info['uses_config'] and not func_info['has_get_config']:
            self.errors.append((
                func_info['name'],
                func_info['lineno'],
                f"Async function '{func_info['name']}' uses 'config' but doesn't call get_config()"
            ))
    
    def visit_Assign(self, node):
        if self.function_stack:
            # Check for config = get_config()
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'config':
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            if node.value.func.id == 'get_config':
                                self.function_stack[-1]['has_get_config'] = True
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if self.function_stack:
            # Check for config.something usage
            if isinstance(node.value, ast.Name) and node.value.id == 'config':
                self.function_stack[-1]['uses_config'] = True
        self.generic_visit(node)


def check_file(file_path: Path) -> List[Tuple[str, int, str]]:
    """Check a Python file for config usage errors."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        checker = ConfigUsageChecker()
        checker.visit(tree)
        return checker.errors
    except SyntaxError as e:
        return [('SyntaxError', e.lineno or 0, str(e))]
    except Exception as e:
        return [('Error', 0, f"Failed to parse {file_path}: {e}")]


def main():
    """Run validation on key files."""
    project_root = Path(__file__).parent.parent
    
    files_to_check = [
        project_root / "run.py",
        project_root / "config.py",
        project_root / "cluster.py",
    ]
    
    all_errors = []
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        errors = check_file(file_path)
        if errors:
            all_errors.extend([(file_path, *error) for error in errors])
    
    if all_errors:
        print("❌ Validation errors found:\n")
        for file_path, func_name, lineno, message in all_errors:
            print(f"  {file_path}:{lineno} in {func_name}")
            print(f"    {message}\n")
        return 1
    else:
        print("✅ All validation checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
