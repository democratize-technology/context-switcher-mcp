#!/usr/bin/env python3
"""
Systematic logging migration script for Context Switcher MCP.

This script migrates all files from old logging patterns to the new standardized
logging interface, improving performance and consistency.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class LoggingMigrator:
    """Handles migration from old logging patterns to standardized logging"""

    def __init__(self, src_path: Path):
        self.src_path = src_path
        self.backup_dir = src_path.parent / "logging_migration_backup"
        self.issues_fixed = []
        self.issues_remaining = []
        
        # Patterns to detect old logging usage
        self.old_patterns = {
            'import_logging': re.compile(r'^import logging$', re.MULTILINE),
            'getlogger_call': re.compile(r'logger = logging\.getLogger\((.*?)\)'),
            'direct_logging': re.compile(r'logging\.(debug|info|warning|error|critical)\('),
            'string_concat_log': re.compile(r'logger\.(debug|info|warning|error|critical)\([^)]*\+[^)]*\)'),
            'fstring_expensive': re.compile(r'logger\.(debug|info|warning|error|critical)\(f["\'][^"\']*\{[^}]+\([^}]+\)[^}]*\}'),
        }

    def create_backup(self) -> None:
        """Create backup of source files before migration"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy entire src directory for backup
        shutil.copytree(self.src_path, self.backup_dir / "src", dirs_exist_ok=True)
        print(f"✓ Created backup at {self.backup_dir}")

    def analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze a file for logging patterns that need migration"""
        issues = {
            'old_imports': [],
            'old_getlogger': [],
            'string_concatenation': [],
            'expensive_operations': [],
            'direct_logging': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for old import patterns
            if self.old_patterns['import_logging'].search(content):
                if 'from .logging_config import' not in content and 'from ..logging_config import' not in content:
                    issues['old_imports'].append("Uses 'import logging' without standardized import")

            # Check for direct getLogger calls
            matches = self.old_patterns['getlogger_call'].findall(content)
            for match in matches:
                issues['old_getlogger'].append(f"Direct logging.getLogger() call: {match}")

            # Check for string concatenation in logs
            if self.old_patterns['string_concat_log'].search(content):
                issues['string_concatenation'].append("String concatenation in log calls")

            # Check for expensive operations in f-strings
            if self.old_patterns['fstring_expensive'].search(content):
                issues['expensive_operations'].append("Expensive operations in f-string logs")

            # Parse AST for more complex analysis
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Check for logging method calls with string concatenation
                        if (isinstance(node.func, ast.Attribute) and 
                            node.func.attr in ['debug', 'info', 'warning', 'error', 'critical']):
                            
                            for arg in node.args:
                                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                                    issues['string_concatenation'].append(f"Line {node.lineno}: String concatenation in log call")
            except SyntaxError:
                pass  # Skip files with syntax errors

        except Exception as e:
            issues['file_error'] = [f"Could not analyze file: {e}"]

        return issues

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file to use standardized logging"""
        if file_path.name.startswith('test_') or file_path.name == '__init__.py':
            return False  # Skip test files and __init__.py
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modifications = []

            # 1. Replace import logging with standardized import
            if re.search(r'^import logging$', content, re.MULTILINE):
                # Check if we need relative import
                relative_path = self._calculate_relative_import(file_path)
                if relative_path:
                    content = re.sub(
                        r'^import logging$',
                        f'from {relative_path}logging_config import get_logger',
                        content,
                        flags=re.MULTILINE
                    )
                    modifications.append("Added standardized logging import")

            # 2. Replace logger = logging.getLogger(__name__) patterns
            getlogger_pattern = re.compile(r'logger = logging\.getLogger\(__name__\)')
            if getlogger_pattern.search(content):
                content = getlogger_pattern.sub('logger = get_logger(__name__)', content)
                modifications.append("Replaced logging.getLogger() with get_logger()")

            # 3. Replace other getLogger patterns
            content = re.sub(
                r'logging\.getLogger\("([^"]+)"\)',
                r'get_logger("\1")',
                content
            )
            content = re.sub(
                r'logging\.getLogger\(([^)]+)\)',
                r'get_logger(\1)',
                content
            )

            # 4. Fix common string concatenation patterns in logs
            # Replace logger.info("message " + var) with logger.info("message %s", var)
            log_methods = ['debug', 'info', 'warning', 'error', 'critical']
            for method in log_methods:
                # Simple string concatenation
                pattern = rf'logger\.{method}\("([^"]*)" \+ ([^,)]+)\)'
                replacement = rf'logger.{method}("\1%s", \2)'
                content = re.sub(pattern, replacement, content)

                # Multiple concatenation
                pattern = rf'logger\.{method}\("([^"]*)" \+ str\(([^)]+)\)\)'
                replacement = rf'logger.{method}("\1%s", \2)'
                content = re.sub(pattern, replacement, content)

            # 5. Fix f-string patterns with expensive operations
            # This is more complex, so we'll add lazy evaluation
            for method in log_methods:
                # Pattern: logger.debug(f"Result: {expensive_function()}")
                pattern = rf'logger\.{method}\(f"([^"]*)\{{([^}}]+\([^}}]+\))}}"'
                if re.search(pattern, content):
                    # For now, add a comment to manually review
                    content = re.sub(
                        pattern,
                        rf'logger.{method}("\1%s", lazy_log(\2))',
                        content
                    )
                    modifications.append(f"Converted f-string with function call to lazy evaluation in {method}")

            # 6. Add import for lazy_log if it was used
            if 'lazy_log(' in content and 'lazy_log' not in content.split('lazy_log(')[0]:
                relative_path = self._calculate_relative_import(file_path)
                if relative_path and 'from ' + relative_path + 'logging_config import' in content:
                    content = content.replace(
                        f'from {relative_path}logging_config import get_logger',
                        f'from {relative_path}logging_config import get_logger, lazy_log'
                    )
                    modifications.append("Added lazy_log import")

            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.issues_fixed.append({
                    'file': str(file_path.relative_to(self.src_path.parent)),
                    'modifications': modifications
                })
                return True

        except Exception as e:
            self.issues_remaining.append(f"{file_path}: Migration error: {e}")

        return False

    def _calculate_relative_import(self, file_path: Path) -> str:
        """Calculate relative import path for logging_config"""
        try:
            # Get relative path from file to src root
            rel_path = file_path.relative_to(self.src_path)
            depth = len(rel_path.parent.parts)
            
            if depth == 0:
                return "."  # Same directory
            else:
                return "." * depth  # Parent directories
        except ValueError:
            return ""  # Fallback

    def migrate_all_files(self) -> None:
        """Migrate all Python files in the source directory"""
        print("🔄 Starting systematic logging migration...\n")
        
        files_processed = 0
        files_migrated = 0
        
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name.startswith('test_'):
                continue  # Skip test files
            
            files_processed += 1
            
            # Analyze file first
            issues = self.analyze_file(py_file)
            has_issues = any(issues.values())
            
            if has_issues:
                print(f"🔧 Migrating: {py_file.relative_to(self.src_path.parent)}")
                
                if self.migrate_file(py_file):
                    files_migrated += 1
                    print("   ✓ Migration completed")
                else:
                    print("   - No changes needed")
            
        print(f"\n📊 Migration Summary:")
        print(f"   Files processed: {files_processed}")
        print(f"   Files migrated: {files_migrated}")
        print(f"   Issues fixed: {len(self.issues_fixed)}")
        print(f"   Issues remaining: {len(self.issues_remaining)}")

    def validate_migration(self) -> List[str]:
        """Validate that migration was successful"""
        print("\n🔍 Validating migration...")
        
        validation_issues = []
        
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name.startswith('test_'):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for remaining old patterns
                if re.search(r'^import logging$', content, re.MULTILINE):
                    if 'from .logging_config import' not in content:
                        validation_issues.append(f"{py_file}: Still has 'import logging' without standardized import")
                
                if 'logging.getLogger(' in content:
                    validation_issues.append(f"{py_file}: Still uses 'logging.getLogger()' directly")
                
                # Check for string concatenation (simple pattern)
                if re.search(r'logger\.(debug|info|warning|error|critical)\([^)]*" \+ [^)]*\)', content):
                    validation_issues.append(f"{py_file}: Still has string concatenation in log calls")
            
            except Exception as e:
                validation_issues.append(f"{py_file}: Validation error: {e}")
        
        return validation_issues

    def run_migration(self) -> None:
        """Run the complete migration process"""
        print("🚀 Context Switcher MCP Logging Migration")
        print("=" * 50)
        
        # Create backup
        self.create_backup()
        
        # Run migration
        self.migrate_all_files()
        
        # Validate results
        validation_issues = self.validate_migration()
        
        if validation_issues:
            print(f"\n⚠️  Validation found {len(validation_issues)} remaining issues:")
            for issue in validation_issues[:10]:  # Show first 10
                print(f"   - {issue}")
            if len(validation_issues) > 10:
                print(f"   ... and {len(validation_issues) - 10} more")
        else:
            print("\n✅ Migration validation successful!")
        
        # Show detailed results
        if self.issues_fixed:
            print(f"\n📝 Files successfully migrated ({len(self.issues_fixed)}):")
            for fix in self.issues_fixed[:5]:  # Show first 5
                print(f"   {fix['file']}: {', '.join(fix['modifications'])}")
            if len(self.issues_fixed) > 5:
                print(f"   ... and {len(self.issues_fixed) - 5} more files")
        
        if self.issues_remaining:
            print(f"\n❌ Issues requiring manual attention ({len(self.issues_remaining)}):")
            for issue in self.issues_remaining[:5]:
                print(f"   {issue}")
        
        print(f"\n💾 Backup created at: {self.backup_dir}")
        print("🔄 Run 'python -m pytest tests/test_unified_logging.py' to validate functionality")


def main():
    """Main entry point"""
    # Get source path
    script_path = Path(__file__).parent
    src_path = script_path / "src" / "context_switcher_mcp"
    
    if not src_path.exists():
        print(f"❌ Source path not found: {src_path}")
        sys.exit(1)
    
    # Run migration
    migrator = LoggingMigrator(src_path)
    migrator.run_migration()


if __name__ == "__main__":
    main()