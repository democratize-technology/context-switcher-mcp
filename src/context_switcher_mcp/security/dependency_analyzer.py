#!/usr/bin/env python3
"""
Dependency Analysis Tool for Context-Switcher MCP
Analyzes import dependencies and identifies circular imports
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import json

from ..logging_config import get_logger

logger = get_logger(__name__)


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import information from Python files"""

    def __init__(self, file_path: str, package_prefix: str = "context_switcher_mcp"):
        self.file_path = file_path
        self.package_prefix = package_prefix
        self.imports = []
        self.relative_imports = []
        self.from_imports = []

    def visit_Import(self, node):
        """Visit regular import statements"""
        for alias in node.names:
            self.imports.append(
                {
                    "type": "import",
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from...import statements"""
        module = node.module or ""
        level = node.level

        for alias in node.names:
            import_info = {
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "level": level,
                "line": node.lineno,
            }

            if level > 0:  # Relative import
                self.relative_imports.append(import_info)
            else:
                self.from_imports.append(import_info)

        self.generic_visit(node)


class DependencyAnalyzer:
    """Analyzes Python project dependencies and identifies circular imports"""

    def __init__(self, project_root: str, package_name: str = "context_switcher_mcp"):
        self.project_root = Path(project_root)
        self.package_name = package_name
        self.package_path = self.project_root / "src" / package_name
        self.module_graph = defaultdict(set)
        self.import_details = {}
        self.file_to_module = {}
        self.module_to_file = {}

    def analyze_project(self) -> Dict:
        """Analyze the entire project for dependencies"""
        logger.info(f"Starting dependency analysis for project at: {self.project_root}")
        logger.info(f"Package path: {self.package_path}")

        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Analyze each file
        for file_path in python_files:
            self._analyze_file(file_path)

        # Build dependency graph
        self._build_dependency_graph()

        # Find circular dependencies
        cycles = self._find_circular_dependencies()

        # Generate analysis report
        return self._generate_report(cycles)

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the package"""
        python_files = []
        if self.package_path.exists():
            for file_path in self.package_path.rglob("*.py"):
                # Skip __pycache__ and test files for now
                if "__pycache__" not in str(file_path):
                    python_files.append(file_path)
        return python_files

    def _analyze_file(self, file_path: Path):
        """Analyze imports in a single file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Create analyzer
            analyzer = ImportAnalyzer(str(file_path), self.package_name)
            analyzer.visit(tree)

            # Convert file path to module name
            module_name = self._file_path_to_module_name(file_path)

            # Store mappings
            self.file_to_module[str(file_path)] = module_name
            self.module_to_file[module_name] = str(file_path)

            # Store import details
            self.import_details[module_name] = {
                "file_path": str(file_path),
                "imports": analyzer.imports,
                "from_imports": analyzer.from_imports,
                "relative_imports": analyzer.relative_imports,
            }

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}", exc_info=True)

    def _file_path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to Python module name"""
        # Get relative path from package root
        try:
            rel_path = file_path.relative_to(self.package_path)
        except ValueError:
            # File is outside package path
            return str(file_path)

        # Convert to module name
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        if not parts:
            return self.package_name

        return f"{self.package_name}.{'.'.join(parts)}"

    def _build_dependency_graph(self):
        """Build the dependency graph from import analysis"""
        for module_name, details in self.import_details.items():
            dependencies = set()

            # Process regular imports
            for imp in details["imports"]:
                if imp["module"].startswith(self.package_name):
                    dependencies.add(imp["module"])

            # Process from imports
            for imp in details["from_imports"]:
                if imp["module"].startswith(self.package_name):
                    dependencies.add(imp["module"])

            # Process relative imports
            for imp in details["relative_imports"]:
                target_module = self._resolve_relative_import(module_name, imp)
                if target_module and target_module.startswith(self.package_name):
                    dependencies.add(target_module)

            self.module_graph[module_name] = dependencies

    def _resolve_relative_import(
        self, current_module: str, import_info: Dict
    ) -> Optional[str]:
        """Resolve relative import to absolute module name"""
        level = import_info["level"]
        module = import_info["module"]

        # Split current module into parts
        current_parts = current_module.split(".")

        # Go up 'level' number of directories
        if level >= len(current_parts):
            return None

        target_parts = current_parts[:-level] if level > 0 else current_parts

        if module:
            target_parts.extend(module.split("."))

        return ".".join(target_parts)

    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependencies using DFS"""

        def dfs(node, path, visited, rec_stack):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                return [path[cycle_start:]]

            if node in visited:
                return []

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            cycles = []
            for neighbor in self.module_graph.get(node, set()):
                cycles.extend(dfs(neighbor, path, visited, rec_stack))

            path.pop()
            rec_stack.remove(node)
            return cycles

        all_cycles = []
        visited = set()

        for module in self.module_graph:
            if module not in visited:
                cycles = dfs(module, [], visited, set())
                all_cycles.extend(cycles)

        # Remove duplicates
        unique_cycles = []
        for cycle in all_cycles:
            normalized = tuple(sorted(cycle))
            if normalized not in [tuple(sorted(c)) for c in unique_cycles]:
                unique_cycles.append(cycle)

        return unique_cycles

    def _generate_report(self, cycles: List[List[str]]) -> Dict:
        """Generate comprehensive dependency analysis report"""
        # Calculate metrics
        total_modules = len(self.module_graph)
        total_dependencies = sum(len(deps) for deps in self.module_graph.values())
        avg_dependencies = (
            total_dependencies / total_modules if total_modules > 0 else 0
        )

        # Find modules with most dependencies
        dependency_counts = {
            module: len(deps) for module, deps in self.module_graph.items()
        }
        most_dependent = sorted(
            dependency_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Find most depended-upon modules
        reverse_deps = defaultdict(set)
        for module, deps in self.module_graph.items():
            for dep in deps:
                reverse_deps[dep].add(module)

        most_depended_upon = sorted(
            [(module, len(dependents)) for module, dependents in reverse_deps.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Detailed import analysis
        import_patterns = self._analyze_import_patterns()

        report = {
            "summary": {
                "total_modules": total_modules,
                "total_dependencies": total_dependencies,
                "average_dependencies_per_module": round(avg_dependencies, 2),
                "circular_dependencies_found": len(cycles),
                "has_circular_dependencies": len(cycles) > 0,
            },
            "circular_dependencies": [
                {
                    "cycle": cycle,
                    "length": len(cycle),
                    "severity": self._assess_cycle_severity(cycle),
                }
                for cycle in cycles
            ],
            "dependency_metrics": {
                "most_dependent_modules": most_dependent,
                "most_depended_upon_modules": most_depended_upon,
            },
            "import_patterns": import_patterns,
            "dependency_graph": {
                module: list(deps) for module, deps in self.module_graph.items()
            },
            "recommendations": self._generate_recommendations(cycles, import_patterns),
        }

        return report

    def _analyze_import_patterns(self) -> Dict:
        """Analyze import patterns for potential issues"""
        patterns = {
            "conditional_imports": [],
            "function_level_imports": [],
            "late_imports": [],
            "circular_import_attempts": [],
        }

        for module_name, details in self.import_details.items():
            file_path = details["file_path"]

            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Look for conditional imports (if statements with imports)
                if re.search(
                    r"if\s+.*:\s*\n\s*(from\s+.*import|import)", content, re.MULTILINE
                ):
                    patterns["conditional_imports"].append(module_name)

                # Look for function-level imports
                if re.search(
                    r"def\s+\w+.*:\s*\n\s*(from\s+.*import|import)",
                    content,
                    re.MULTILINE,
                ):
                    patterns["function_level_imports"].append(module_name)

                # Look for late imports (imports after line 50)
                lines = content.split("\n")
                for i, line in enumerate(lines[50:], 51):
                    if re.match(r"\s*(from\s+.*import|import)", line):
                        patterns["late_imports"].append(
                            {"module": module_name, "line": i, "import": line.strip()}
                        )
                        break

            except Exception as e:
                logger.error(
                    f"Error analyzing patterns in {file_path}: {e}", exc_info=True
                )

        return patterns

    def _assess_cycle_severity(self, cycle: List[str]) -> str:
        """Assess the severity of a circular dependency"""
        if len(cycle) == 2:
            return "HIGH"  # Direct circular dependency
        elif len(cycle) <= 4:
            return "MEDIUM"  # Short cycle
        else:
            return "LOW"  # Long cycle, might be manageable

    def _generate_recommendations(
        self, cycles: List[List[str]], patterns: Dict
    ) -> List[str]:
        """Generate recommendations for fixing dependency issues"""
        recommendations = []

        if cycles:
            recommendations.append(
                "CRITICAL: Circular dependencies found - these must be resolved"
            )
            recommendations.append(
                "Consider using dependency injection or factory patterns"
            )
            recommendations.append("Move shared types to separate modules")
            recommendations.append("Use protocol/interface modules for contracts")

        if patterns["conditional_imports"]:
            recommendations.append(
                "Consider refactoring conditional imports to use dependency injection"
            )

        if patterns["function_level_imports"]:
            recommendations.append(
                "Function-level imports found - consider moving to module level with TYPE_CHECKING"
            )

        if patterns["late_imports"]:
            recommendations.append(
                "Late imports found - consider reorganizing module dependencies"
            )

        return recommendations


def main():
    """Main analysis function"""
    # Get the project root (parent of src directory)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent  # Go up to project root

    logger.info("Starting comprehensive dependency analysis")
    logger.debug(f"Analysis script: {__file__}")
    logger.debug(f"Project root: {project_root}")

    analyzer = DependencyAnalyzer(str(project_root))
    report = analyzer.analyze_project()

    # Save report to file
    report_file = current_dir / "dependency_analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=== DEPENDENCY ANALYSIS REPORT ===")
    logger.info(f"Total modules: {report['summary']['total_modules']}")
    logger.info(f"Total dependencies: {report['summary']['total_dependencies']}")
    logger.info(
        f"Average dependencies per module: {report['summary']['average_dependencies_per_module']:.2f}"
    )
    logger.warning(
        f"Circular dependencies found: {report['summary']['circular_dependencies_found']}"
    )

    if report["circular_dependencies"]:
        logger.warning("=== CIRCULAR DEPENDENCIES DETECTED ===")
        for i, cycle_info in enumerate(report["circular_dependencies"], 1):
            cycle = cycle_info["cycle"]
            severity = cycle_info["severity"]
            logger.error(
                f"Circular dependency {i}: {severity} severity - Length {len(cycle)}"
            )
            logger.error(f"   Cycle: {' -> '.join(cycle)} -> {cycle[0]}")

    logger.info("=== RECOMMENDATIONS ===")
    for rec in report["recommendations"]:
        logger.info(f"- {rec}")

    logger.info(f"Full dependency analysis report saved to: {report_file}")

    return report


if __name__ == "__main__":
    main()
