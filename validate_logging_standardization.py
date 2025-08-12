#!/usr/bin/env python3
"""
Validation script for logging standardization compliance.

This script validates that all files in the Context Switcher MCP project
follow the standardized logging patterns and identifies any remaining issues.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LoggingIssue:
    """Represents a logging standardization issue"""
    file_path: Path
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggested_fix: str


class LoggingStandardizationValidator:
    """Validates logging standardization across the codebase"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src" / "context_switcher_mcp"
        self.issues: List[LoggingIssue] = []
        self.stats = {
            'files_analyzed': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'issues_by_type': defaultdict(int),
            'issues_by_severity': defaultdict(int)
        }
        
        # Patterns for detecting issues
        self.old_patterns = {
            'import_logging_only': re.compile(r'^import logging$', re.MULTILINE),
            'direct_getlogger': re.compile(r'logger\s*=\s*logging\.getLogger\([^)]*\)'),
            'string_concat_log': re.compile(r'logger\.(debug|info|warning|error|critical)\([^)]*"[^"]*"\s*\+[^)]*\)'),
            'fstring_expensive': re.compile(r'logger\.(debug|info|warning|error|critical)\(f["\'][^"\']*\{[^}]+\([^}]+\)[^}]*\}'),
            'direct_logging_module': re.compile(r'logging\.(debug|info|warning|error|critical)\('),
        }
        
        # Good patterns to look for
        self.good_patterns = {
            'standardized_import': re.compile(r'from \.+logging_config import get_logger'),
            'get_logger_usage': re.compile(r'get_logger\(__name__\)|get_logger\("[^"]+"\)'),
            'lazy_log_usage': re.compile(r'lazy_log\('),
            'structured_logging': re.compile(r'log_structured\('),
        }

    def analyze_file(self, file_path: Path) -> List[LoggingIssue]:
        """Analyze a single file for logging compliance"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=0,
                issue_type='file_error',
                description=f"Could not read file: {e}",
                severity='error',
                suggested_fix="Ensure file is readable and properly encoded"
            ))
            return issues
        
        lines = content.splitlines()
        
        # Check for old import patterns
        if self.old_patterns['import_logging_only'].search(content):
            # Check if it also has the new standardized import
            if not self.good_patterns['standardized_import'].search(content):
                line_num = self._find_line_number(lines, r'import logging')
                issues.append(LoggingIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type='old_import',
                    description="Uses 'import logging' without standardized import",
                    severity='error',
                    suggested_fix="Add: from .logging_config import get_logger"
                ))
        
        # Check for direct getLogger usage
        for match in self.old_patterns['direct_getlogger'].finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=line_num,
                issue_type='direct_getlogger',
                description="Uses logging.getLogger() directly",
                severity='error',
                suggested_fix="Replace with: logger = get_logger(__name__)"
            ))
        
        # Check for string concatenation in log calls
        for match in self.old_patterns['string_concat_log'].finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=line_num,
                issue_type='string_concatenation',
                description="String concatenation in log call (performance issue)",
                severity='warning',
                suggested_fix="Use parameter substitution: logger.info('Message %s', var)"
            ))
        
        # Check for expensive f-string operations
        for match in self.old_patterns['fstring_expensive'].finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=line_num,
                issue_type='expensive_fstring',
                description="Expensive operation in f-string log (performance issue)",
                severity='warning',
                suggested_fix="Use lazy_log: logger.debug('Result: %s', lazy_log(expensive_func))"
            ))
        
        # Check for direct logging module usage
        for match in self.old_patterns['direct_logging_module'].finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=line_num,
                issue_type='direct_logging_module',
                description="Uses logging module directly instead of logger instance",
                severity='error',
                suggested_fix="Use logger instance: logger.info() instead of logging.info()"
            ))
        
        # AST-based analysis for more complex patterns
        try:
            tree = ast.parse(content)
            ast_issues = self._analyze_ast(tree, file_path, lines)
            issues.extend(ast_issues)
        except SyntaxError as e:
            issues.append(LoggingIssue(
                file_path=file_path,
                line_number=e.lineno or 0,
                issue_type='syntax_error',
                description=f"Syntax error prevents analysis: {e}",
                severity='error',
                suggested_fix="Fix syntax error"
            ))
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[LoggingIssue]:
        """Perform AST-based analysis for complex patterns"""
        issues = []
        
        for node in ast.walk(tree):
            # Check for string concatenation in log method calls
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['debug', 'info', 'warning', 'error', 'critical']):
                    
                    # Check arguments for string concatenation
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                            issues.append(LoggingIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                issue_type='ast_string_concat',
                                description="String concatenation detected in log call",
                                severity='warning',
                                suggested_fix="Use parameter substitution or structured logging"
                            ))
                        
                        # Check for function calls in f-strings (more complex analysis)
                        elif isinstance(arg, ast.JoinedStr):
                            for value in arg.values:
                                if isinstance(value, ast.FormattedValue):
                                    if isinstance(value.value, ast.Call):
                                        issues.append(LoggingIssue(
                                            file_path=file_path,
                                            line_number=node.lineno,
                                            issue_type='fstring_function_call',
                                            description="Function call in f-string log (may impact performance)",
                                            severity='info',
                                            suggested_fix="Consider lazy_log for expensive operations"
                                        ))
        
        return issues
    
    def _find_line_number(self, lines: List[str], pattern: str) -> int:
        """Find line number of a pattern in file lines"""
        compiled_pattern = re.compile(pattern)
        for i, line in enumerate(lines, 1):
            if compiled_pattern.search(line):
                return i
        return 0
    
    def validate_project(self) -> None:
        """Validate the entire project for logging standardization"""
        print("🔍 Validating logging standardization across Context Switcher MCP...")
        print("=" * 70)
        
        # Find all Python files
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.src_path.rglob(pattern))
        
        # Filter out test files and __pycache__
        python_files = [
            f for f in python_files 
            if not f.name.startswith('test_') and '__pycache__' not in str(f)
        ]
        
        print(f"📁 Found {len(python_files)} Python files to analyze")
        print()
        
        # Analyze each file
        for file_path in python_files:
            self.stats['files_analyzed'] += 1
            
            relative_path = file_path.relative_to(self.project_root)
            print(f"Analyzing: {relative_path}")
            
            file_issues = self.analyze_file(file_path)
            if file_issues:
                self.stats['files_with_issues'] += 1
                self.issues.extend(file_issues)
                
                # Update statistics
                for issue in file_issues:
                    self.stats['total_issues'] += 1
                    self.stats['issues_by_type'][issue.issue_type] += 1
                    self.stats['issues_by_severity'][issue.severity] += 1
        
        print(f"\n✅ Analysis complete: {self.stats['files_analyzed']} files analyzed")
        print()
    
    def print_summary_report(self) -> None:
        """Print a summary report of validation results"""
        print("📊 LOGGING STANDARDIZATION VALIDATION REPORT")
        print("=" * 50)
        print()
        
        # Overall statistics
        print("📈 Overall Statistics:")
        print(f"   Files analyzed:      {self.stats['files_analyzed']}")
        print(f"   Files with issues:   {self.stats['files_with_issues']}")
        print(f"   Total issues found:  {self.stats['total_issues']}")
        print()
        
        # Issues by severity
        if self.stats['total_issues'] > 0:
            print("🚨 Issues by Severity:")
            for severity, count in sorted(self.stats['issues_by_severity'].items()):
                print(f"   {severity.upper():<8}: {count}")
            print()
            
            # Issues by type
            print("🔍 Issues by Type:")
            for issue_type, count in sorted(self.stats['issues_by_type'].items()):
                print(f"   {issue_type:<20}: {count}")
            print()
        
        # Compliance score
        if self.stats['files_analyzed'] > 0:
            compliant_files = self.stats['files_analyzed'] - self.stats['files_with_issues']
            compliance_rate = (compliant_files / self.stats['files_analyzed']) * 100
            
            print(f"✅ Compliance Rate: {compliance_rate:.1f}%")
            print(f"   ({compliant_files}/{self.stats['files_analyzed']} files compliant)")
        
        print()
    
    def print_detailed_report(self, max_issues_per_type: int = 5) -> None:
        """Print detailed report of all issues found"""
        if not self.issues:
            print("🎉 No logging standardization issues found!")
            return
        
        print("📋 DETAILED ISSUES REPORT")
        print("=" * 50)
        print()
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append(issue)
        
        for issue_type, issues in sorted(issues_by_type.items()):
            print(f"🔴 {issue_type.upper().replace('_', ' ')} ({len(issues)} issues)")
            print("-" * 40)
            
            # Show sample issues
            for i, issue in enumerate(issues[:max_issues_per_type]):
                relative_path = issue.file_path.relative_to(self.project_root)
                print(f"   {i+1}. {relative_path}:{issue.line_number}")
                print(f"      Description: {issue.description}")
                print(f"      Severity: {issue.severity}")
                print(f"      Fix: {issue.suggested_fix}")
                print()
            
            if len(issues) > max_issues_per_type:
                print(f"   ... and {len(issues) - max_issues_per_type} more similar issues")
                print()
    
    def generate_migration_plan(self) -> None:
        """Generate a migration plan based on found issues"""
        if not self.issues:
            print("✅ No migration needed - all files are compliant!")
            return
        
        print("📋 MIGRATION PLAN")
        print("=" * 30)
        print()
        
        # Group by severity and provide actionable steps
        error_issues = [i for i in self.issues if i.severity == 'error']
        warning_issues = [i for i in self.issues if i.severity == 'warning']
        info_issues = [i for i in self.issues if i.severity == 'info']
        
        print("🔥 PRIORITY 1 - Critical Issues (Must Fix)")
        if error_issues:
            error_files = set(i.file_path for i in error_issues)
            print(f"   Affects {len(error_files)} files with {len(error_issues)} issues")
            print("   Action: Run logging_migration.py to automatically fix most issues")
        else:
            print("   ✅ No critical issues found")
        print()
        
        print("⚠️  PRIORITY 2 - Performance Issues (Should Fix)")  
        if warning_issues:
            warning_files = set(i.file_path for i in warning_issues)
            print(f"   Affects {len(warning_files)} files with {len(warning_issues)} issues")
            print("   Action: Manually review and optimize performance-critical log calls")
        else:
            print("   ✅ No performance issues found")
        print()
        
        print("💡 PRIORITY 3 - Improvements (Nice to Have)")
        if info_issues:
            info_files = set(i.file_path for i in info_issues)
            print(f"   Affects {len(info_files)} files with {len(info_issues)} issues")
            print("   Action: Consider optimizations during regular code review")
        else:
            print("   ✅ No improvement opportunities found")
        print()
        
        # Recommended tools
        print("🛠️  RECOMMENDED TOOLS:")
        print("   1. Run: python logging_migration.py")
        print("   2. Run: python -m pytest tests/test_unified_logging.py")
        print("   3. Run: python validate_logging_standardization.py")
        print()
    
    def export_json_report(self, output_file: Path) -> None:
        """Export detailed report as JSON for CI/CD integration"""
        import json
        
        report_data = {
            'validation_timestamp': str(Path(__file__).stat().st_mtime),
            'project_root': str(self.project_root),
            'statistics': dict(self.stats),
            'issues': [
                {
                    'file_path': str(issue.file_path.relative_to(self.project_root)),
                    'line_number': issue.line_number,
                    'issue_type': issue.issue_type,
                    'description': issue.description,
                    'severity': issue.severity,
                    'suggested_fix': issue.suggested_fix
                }
                for issue in self.issues
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 JSON report exported to: {output_file}")


def main():
    """Main entry point"""
    # Get project root
    script_path = Path(__file__).parent
    project_root = script_path
    
    if not (project_root / "src" / "context_switcher_mcp").exists():
        print(f"❌ Could not find Context Switcher MCP source at: {project_root}")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Validate logging standardization")
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed report of all issues')
    parser.add_argument('--json', type=Path,
                       help='Export JSON report to specified file')
    parser.add_argument('--max-issues', type=int, default=5,
                       help='Maximum issues to show per type in detailed report')
    
    args = parser.parse_args()
    
    # Run validation
    validator = LoggingStandardizationValidator(project_root)
    validator.validate_project()
    
    # Generate reports
    validator.print_summary_report()
    
    if args.detailed or validator.stats['total_issues'] <= 20:
        validator.print_detailed_report(args.max_issues)
    
    validator.generate_migration_plan()
    
    if args.json:
        validator.export_json_report(args.json)
    
    # Exit with error code if issues found
    if validator.stats['total_issues'] > 0:
        sys.exit(1)
    else:
        print("🎉 All files are compliant with logging standards!")
        sys.exit(0)


if __name__ == "__main__":
    main()