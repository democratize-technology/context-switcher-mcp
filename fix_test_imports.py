#!/usr/bin/env python3
"""
Script to systematically fix incorrect patch paths in test files
"""

import re
from pathlib import Path


def fix_test_imports():
    """Fix incorrect patch paths in test files"""

    # Mapping of incorrect patch paths to correct ones
    replacements = {
        # session_manager is always imported from parent module
        r"context_switcher_mcp\.tools\..*?\.session_manager": "context_switcher_mcp.session_manager",
        # orchestrator is always from parent module
        r"context_switcher_mcp\.tools\..*?\.orchestrator": "context_switcher_mcp.orchestrator",
        # PerspectiveOrchestrator is from perspective_orchestrator module
        r"context_switcher_mcp\.tools\..*?\.PerspectiveOrchestrator": "context_switcher_mcp.perspective_orchestrator.PerspectiveOrchestrator",
        # Helper functions are from their actual modules
        r"context_switcher_mcp\.tools\..*?\.validate_analysis_request": "context_switcher_mcp.helpers.analysis_helpers.validate_analysis_request",
        r"context_switcher_mcp\.tools\..*?\.build_analysis_aorp_response": "context_switcher_mcp.helpers.analysis_helpers.build_analysis_aorp_response",
        # AORP functions are from aorp module
        r"context_switcher_mcp\.tools\..*?\.create_error_response": "context_switcher_mcp.aorp.create_error_response",
        r"context_switcher_mcp\.tools\..*?\.generate_synthesis_next_steps": "context_switcher_mcp.aorp.generate_synthesis_next_steps",
        # Validation functions
        r"context_switcher_mcp\.tools\..*?\.validate_session_id": "context_switcher_mcp.validation.validate_session_id",
        # Response formatter - need to check where this is from
        r"context_switcher_mcp\.tools\..*?\.ResponseFormatter": "context_switcher_mcp.response_formatter.ResponseFormatter",
        r"context_switcher_mcp\.tools\..*?\.calculate_synthesis_confidence": "context_switcher_mcp.response_formatter.calculate_synthesis_confidence",
    }

    # Find all test files
    test_dir = Path("tests")
    test_files = list(test_dir.rglob("*.py"))

    for test_file in test_files:
        print(f"Processing {test_file}")

        try:
            with open(test_file) as f:
                content = f.read()

            original_content = content

            # Apply all replacements
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)

            # Write back if changed
            if content != original_content:
                with open(test_file, "w") as f:
                    f.write(content)
                print(f"  ✓ Fixed imports in {test_file}")
            else:
                print(f"  - No changes needed in {test_file}")

        except Exception as e:
            print(f"  ✗ Error processing {test_file}: {e}")


if __name__ == "__main__":
    fix_test_imports()
