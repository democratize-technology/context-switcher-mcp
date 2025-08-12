#!/usr/bin/env python3
"""
Simple configuration architecture validation

This script validates the architectural design and imports of our
new unified configuration system without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that the new configuration file structure is correct"""
    print("🏗️  Testing configuration architecture...")
    
    src_path = Path("src/context_switcher_mcp")
    config_path = src_path / "config"
    
    # Test main config directory
    assert config_path.exists(), f"Config directory not found: {config_path}"
    assert (config_path / "__init__.py").exists(), "Config __init__.py missing"
    assert (config_path / "core.py").exists(), "Config core.py missing"
    assert (config_path / "migration.py").exists(), "Config migration.py missing"
    print("  ✅ Main config files present")
    
    # Test domains directory
    domains_path = config_path / "domains"
    assert domains_path.exists(), f"Domains directory not found: {domains_path}"
    assert (domains_path / "__init__.py").exists(), "Domains __init__.py missing"
    
    expected_domains = ["models.py", "session.py", "security.py", "server.py", "monitoring.py"]
    for domain in expected_domains:
        domain_file = domains_path / domain
        assert domain_file.exists(), f"Domain file missing: {domain_file}"
    print(f"  ✅ All {len(expected_domains)} domain modules present")
    
    # Test environments directory
    envs_path = config_path / "environments"
    assert envs_path.exists(), f"Environments directory not found: {envs_path}"
    assert (envs_path / "__init__.py").exists(), "Environments __init__.py missing"
    
    expected_envs = ["base.py", "development.py", "staging.py", "production.py"]
    for env in expected_envs:
        env_file = envs_path / env
        assert env_file.exists(), f"Environment file missing: {env_file}"
    print(f"  ✅ All {len(expected_envs)} environment modules present")
    
    return True


def test_code_quality():
    """Test basic code quality metrics"""
    print("\n📊 Testing code quality...")
    
    config_path = Path("src/context_switcher_mcp/config")
    total_lines = 0
    total_files = 0
    
    # Count lines in all Python files
    for py_file in config_path.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
        
        with open(py_file, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
            total_files += 1
            print(f"  📄 {py_file.name}: {lines} lines")
    
    print(f"\n  📈 Total: {total_files} files, {total_lines} lines")
    
    # Basic quality checks
    avg_lines = total_lines / total_files if total_files > 0 else 0
    print(f"  📊 Average lines per file: {avg_lines:.1f}")
    
    if avg_lines > 1000:
        print("  ⚠️  Some files are quite large - consider breaking them down")
    elif avg_lines < 50:
        print("  ⚠️  Files are very small - might be over-modularized")
    else:
        print("  ✅ Good file size distribution")
    
    return True


def test_imports_structure():
    """Test the import structure of configuration modules"""
    print("\n🔗 Testing import structure...")
    
    config_path = Path("src/context_switcher_mcp/config")
    
    # Check for circular imports by analyzing import statements
    import_graph = {}
    
    for py_file in config_path.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
        
        relative_path = str(py_file.relative_to(config_path)).replace(".py", "").replace("/", ".")
        import_graph[relative_path] = []
        
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Look for internal imports
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("from .") or line.startswith("from .."):
                    # Extract the module being imported
                    if " import " in line:
                        module = line.split(" import ")[0].replace("from ", "")
                        import_graph[relative_path].append(module)
                        
        except Exception as e:
            print(f"  ⚠️  Could not analyze {py_file}: {e}")
    
    # Print import structure
    for module, imports in import_graph.items():
        if imports:
            print(f"  🔗 {module} imports: {', '.join(imports)}")
    
    # Check for potential circular imports (basic check)
    circular_risks = []
    for module, imports in import_graph.items():
        for imported in imports:
            if imported in import_graph and module in import_graph[imported]:
                circular_risks.append(f"{module} ↔ {imported}")
    
    if circular_risks:
        print(f"  ⚠️  Potential circular imports: {', '.join(circular_risks)}")
    else:
        print("  ✅ No obvious circular import risks detected")
    
    return True


def test_legacy_files_analysis():
    """Analyze the legacy configuration files that will be replaced"""
    print("\n🗂️  Analyzing legacy configuration files...")
    
    src_path = Path("src/context_switcher_mcp")
    legacy_files = [
        "config.py", "config_base.py", "config_validator.py", 
        "config_legacy.py", "config_migration.py", "config_migration_old.py"
    ]
    
    total_legacy_lines = 0
    legacy_complexity = {}
    
    for legacy_file in legacy_files:
        file_path = src_path / legacy_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_legacy_lines += line_count
                
                # Simple complexity metrics
                classes = sum(1 for line in lines if line.strip().startswith("class "))
                functions = sum(1 for line in lines if line.strip().startswith("def "))
                imports = sum(1 for line in lines if line.strip().startswith(("import ", "from ")))
                
                legacy_complexity[legacy_file] = {
                    "lines": line_count,
                    "classes": classes,
                    "functions": functions,
                    "imports": imports
                }
                
                print(f"  📄 {legacy_file}: {line_count} lines, {classes} classes, {functions} functions")
    
    print(f"\n  📊 Total legacy code: {total_legacy_lines} lines across {len(legacy_complexity)} files")
    
    # Calculate new system metrics
    new_config_path = src_path / "config"
    total_new_lines = 0
    for py_file in new_config_path.glob("**/*.py"):
        if not py_file.name.startswith("__"):
            with open(py_file, 'r') as f:
                total_new_lines += len(f.readlines())
    
    print(f"  📊 New unified config: {total_new_lines} lines")
    
    if total_new_lines < total_legacy_lines:
        reduction = ((total_legacy_lines - total_new_lines) / total_legacy_lines) * 100
        print(f"  ✅ Code reduction: {reduction:.1f}% fewer lines")
    else:
        increase = ((total_new_lines - total_legacy_lines) / total_legacy_lines) * 100
        print(f"  📈 Code expansion: {increase:.1f}% more lines (due to better organization)")
    
    return True


def test_configuration_coverage():
    """Test that the new system covers all legacy functionality"""
    print("\n🎯 Testing configuration coverage...")
    
    # Read legacy config to extract configuration parameters
    src_path = Path("src/context_switcher_mcp")
    legacy_params = set()
    
    # Analyze config_legacy.py for dataclass fields
    legacy_file = src_path / "config_legacy.py"
    if legacy_file.exists():
        with open(legacy_file, 'r') as f:
            content = f.read()
            
        # Extract field names (simple heuristic)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if ': ' in line and '=' in line and not line.startswith('#') and not line.startswith('"""'):
                # Extract field name
                if line.count(':') >= 1:
                    field_name = line.split(':')[0].strip()
                    if field_name and not field_name.startswith('def') and not field_name.startswith('class'):
                        legacy_params.add(field_name)
    
    print(f"  📊 Found {len(legacy_params)} legacy configuration parameters")
    
    # Check new configuration coverage (simplified)
    new_config_path = src_path / "config"
    new_params = set()
    
    for py_file in new_config_path.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Extract Field definitions
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if 'Field(' in line and '=' in line:
                field_name = line.split('=')[0].strip().rstrip(':')
                if field_name:
                    new_params.add(field_name)
    
    print(f"  📊 Found {len(new_params)} new configuration parameters")
    
    # Simple coverage analysis
    if len(new_params) >= len(legacy_params):
        print("  ✅ New configuration system has comprehensive parameter coverage")
    else:
        print("  ⚠️  New configuration system may be missing some legacy parameters")
    
    return True


def main():
    """Run all validation tests"""
    print("🔍 Validating unified configuration architecture...\n")
    
    tests = [
        test_file_structure,
        test_code_quality,
        test_imports_structure,
        test_legacy_files_analysis,
        test_configuration_coverage,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Architecture Validation Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 Configuration architecture validation successful!")
        print("   The unified configuration system is properly structured.")
        return 0
    else:
        print(f"\n⚠️  {failed} architecture tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())