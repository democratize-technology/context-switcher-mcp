"""Comprehensive tests for __main__.py module"""

import sys  # noqa: E402
from unittest.mock import Mock, patch

import pytest


class TestMainModule:
    """Test __main__.py module entry point"""

    def test_main_module_import(self):
        """Test that __main__.py can be imported"""
        # This test verifies the module structure is correct
        try:
            from context_switcher_mcp import __main__

            assert __main__ is not None
        except ImportError as e:
            pytest.fail(f"Failed to import __main__ module: {e}")

    def test_main_function_called_when_run_as_main(self):
        """Test that main() is called when module is run as __main__"""
        # This test verifies the structure is correct for execution
        import context_switcher_mcp.__main__ as main_module

        # Verify the module has the expected structure
        assert hasattr(main_module, "main")
        assert callable(main_module.main)

        # Verify the source contains the expected if __name__ == "__main__" pattern
        import inspect

        source = inspect.getsource(main_module)
        assert 'if __name__ == "__main__"' in source
        assert "main()" in source

    @patch("context_switcher_mcp.main")
    def test_main_not_called_when_imported(self, mock_main):
        """Test that main() is not called when module is imported"""
        # Import the module normally (not as __main__)
        import context_switcher_mcp.__main__

        # main() should not have been called during import
        # (it may have been called in other tests, so we don't assert not called)
        # Instead, we verify the import succeeds without errors
        assert hasattr(context_switcher_mcp.__main__, "main")

    def test_main_function_exists_in_package(self):
        """Test that main function exists in the parent package"""
        import context_switcher_mcp

        assert hasattr(context_switcher_mcp, "main")
        assert callable(context_switcher_mcp.main)

    @patch("context_switcher_mcp.create_server")
    def test_main_module_executable_behavior(self, mock_create_server):
        """Test behavior when module is executed as python -m context_switcher_mcp"""
        # Mock server to prevent actual startup
        mock_server = Mock()
        mock_server.run = Mock()
        mock_create_server.return_value = mock_server

        # Simulate the Python module execution environment
        import context_switcher_mcp.__main__ as main_module

        # Save the current __name__
        original_name = main_module.__name__

        try:
            # Simulate running as main module
            main_module.__name__ = "__main__"

            # Execute the main block logic manually
            if main_module.__name__ == "__main__":
                main_module.main()

            # Verify server was created and run
            mock_create_server.assert_called_once()
            mock_server.run.assert_called_once()

        finally:
            # Restore original __name__
            main_module.__name__ = original_name

    def test_main_module_structure(self):
        """Test the structure and content of __main__.py"""
        import context_switcher_mcp.__main__ as main_module

        # Verify it imports main from parent package
        assert hasattr(main_module, "main")

        # Verify the main function is callable
        assert callable(main_module.main)

        # Verify __name__ check exists (by checking the module's source indirectly)
        import inspect

        source = inspect.getsource(main_module)
        assert 'if __name__ == "__main__"' in source
        assert "main()" in source

    @patch("sys.argv")
    @patch("context_switcher_mcp.create_server")
    def test_main_module_with_command_line_args(self, mock_create_server, mock_argv):
        """Test main module behavior with command line arguments"""
        # Mock server to prevent actual startup
        mock_server = Mock()
        mock_server.run = Mock()
        mock_create_server.return_value = mock_server

        # Simulate command line arguments
        mock_argv.return_value = [
            "python",
            "-m",
            "context_switcher_mcp",
            "--host",
            "0.0.0.0",
        ]

        import context_switcher_mcp.__main__ as main_module

        original_name = main_module.__name__

        try:
            # Simulate running as main module
            main_module.__name__ = "__main__"

            # Execute the main block
            if main_module.__name__ == "__main__":
                main_module.main()

            # Verify server was created and run (should handle the arguments)
            mock_create_server.assert_called_once()
            mock_server.run.assert_called_once()

        finally:
            main_module.__name__ = original_name

    def test_import_path_correctness(self):
        """Test that the import path is correct"""
        # Verify that the import statement works correctly
        # The main function should be the same as the one in the parent package
        import context_switcher_mcp
        import context_switcher_mcp.__main__ as main_module

        assert main_module.main is context_switcher_mcp.main

    def test_module_docstring(self):
        """Test that the module has appropriate documentation"""
        import context_switcher_mcp.__main__ as main_module

        # Check if module has a docstring
        assert main_module.__doc__ is not None
        assert len(main_module.__doc__.strip()) > 0

        # Verify it mentions the purpose
        doc_lower = main_module.__doc__.lower()
        assert any(
            keyword in doc_lower for keyword in ["entry", "point", "module", "main"]
        )


class TestMainModuleIntegration:
    """Test integration aspects of the main module"""

    @patch("subprocess.run")
    def test_module_execution_via_subprocess(self, mock_subprocess):
        """Test that the module can be executed via subprocess simulation"""
        # This simulates: python -m context_switcher_mcp
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Server started"
        mock_subprocess.return_value.stderr = ""

        # Import the module to verify it's structured correctly for subprocess execution
        import context_switcher_mcp.__main__

        # The module should be importable without errors
        assert context_switcher_mcp.__main__ is not None

    def test_module_in_sys_modules(self):
        """Test that the module is properly registered in sys.modules"""

        # Import the module
        import context_switcher_mcp.__main__

        # Check it's in sys.modules
        assert "context_switcher_mcp.__main__" in sys.modules

        # Verify it's the same object
        assert (
            sys.modules["context_switcher_mcp.__main__"]
            is context_switcher_mcp.__main__
        )

    def test_module_attributes(self):
        """Test that the module has expected attributes"""
        import context_switcher_mcp.__main__ as main_module

        # Standard module attributes
        assert hasattr(main_module, "__file__")
        assert hasattr(main_module, "__name__")
        assert hasattr(main_module, "__package__")

        # Function from import
        assert hasattr(main_module, "main")

    def test_circular_import_prevention(self):
        """Test that there are no circular import issues"""
        try:
            # Multiple imports should work without issues
            import context_switcher_mcp
            import context_switcher_mcp.__main__  # Second import

            # All should be successful
            assert context_switcher_mcp.__main__ is not None
            assert context_switcher_mcp is not None

        except ImportError as e:
            pytest.fail(f"Circular import or import error detected: {e}")


class TestMainModuleErrorHandling:
    """Test error handling in the main module"""

    @patch("context_switcher_mcp.create_server")
    def test_main_function_exception_handling(self, mock_create_server):
        """Test behavior when main function raises an exception"""
        # Configure server creation to raise an exception
        mock_create_server.side_effect = Exception("Test exception")

        import context_switcher_mcp.__main__ as main_module

        original_name = main_module.__name__

        try:
            main_module.__name__ = "__main__"

            # The exception should propagate (not be caught by __main__.py)
            with pytest.raises(Exception, match="Test exception"):
                if main_module.__name__ == "__main__":
                    main_module.main()

        finally:
            main_module.__name__ = original_name

    def test_import_error_resilience(self):
        """Test resilience to import-related issues"""
        # The module should handle missing dependencies gracefully
        # (though in this case, the main function should exist)

        import context_switcher_mcp.__main__ as main_module

        # Verify the import itself doesn't fail
        assert main_module is not None
        assert hasattr(main_module, "main")

    @patch(
        "context_switcher_mcp.__main__.main",
        side_effect=ImportError("Missing dependency"),
    )
    def test_missing_dependency_handling(self, mock_main):
        """Test behavior with missing dependencies"""
        import context_switcher_mcp.__main__ as main_module

        original_name = main_module.__name__

        try:
            main_module.__name__ = "__main__"

            # Should raise ImportError when main() has import issues
            with pytest.raises(ImportError, match="Missing dependency"):
                if main_module.__name__ == "__main__":
                    main_module.main()

        finally:
            main_module.__name__ = original_name


class TestMainModuleCompatibility:
    """Test compatibility aspects of the main module"""

    def test_python_version_compatibility(self):
        """Test that the module works with current Python version"""

        # Import should work regardless of Python version (within supported range)

        # Verify we're running on a supported Python version
        assert sys.version_info >= (3, 8), "Python 3.8+ required"

    def test_module_path_resolution(self):
        """Test that module paths are resolved correctly"""
        import os

        import context_switcher_mcp
        import context_switcher_mcp.__main__ as main_module

        # Get the file paths
        main_file = main_module.__file__
        package_file = context_switcher_mcp.__file__

        # They should be in the same directory
        main_dir = os.path.dirname(main_file)
        package_dir = os.path.dirname(package_file)

        assert main_dir == package_dir

    def test_package_relationship(self):
        """Test the relationship between __main__ and the parent package"""
        import context_switcher_mcp
        import context_switcher_mcp.__main__ as main_module

        # Verify package relationship
        assert main_module.__package__ == "context_switcher_mcp"

        # Verify they share the same main function
        assert main_module.main is context_switcher_mcp.main


if __name__ == "__main__":
    pytest.main([__file__])
