"""Configuration validation utilities and error reporting tools

This module provides utilities for validating configuration files, generating
reports, and diagnosing configuration issues. It's designed to help developers
and operators ensure their configuration is correct before deployment.
"""

import json
import os
import sys
from typing import Any

from .config import ContextSwitcherConfig as LegacyConfig
from .config_migration import (
    generate_migration_report,
)
from .logging_base import get_logger
from .validated_config import (
    ConfigurationError,
    load_validated_config,
)

logger = get_logger(__name__)


class ConfigurationValidator:
    """Comprehensive configuration validator and reporting tool"""

    def __init__(self, config_file: str | None = None):
        """Initialize validator

        Args:
            config_file: Optional path to configuration file to validate
        """
        self.config_file = config_file
        self.validation_results: dict[str, Any] = {}

    def validate_all(self) -> dict[str, Any]:
        """Run comprehensive validation suite

        Returns:
            Dictionary containing validation results
        """
        results = {
            "timestamp": self._get_timestamp(),
            "config_file": self.config_file,
            "validation_summary": {},
            "detailed_results": {},
            "recommendations": [],
        }

        # 1. Validate configuration loading
        load_result = self._validate_config_loading()
        results["detailed_results"]["loading"] = load_result
        results["validation_summary"]["loading_success"] = load_result["success"]

        # 2. Validate individual parameters
        param_result = self._validate_parameters()
        results["detailed_results"]["parameters"] = param_result
        results["validation_summary"]["parameter_issues"] = len(param_result["issues"])

        # 3. Validate cross-parameter constraints
        constraint_result = self._validate_constraints()
        results["detailed_results"]["constraints"] = constraint_result
        results["validation_summary"]["constraint_issues"] = len(
            constraint_result["issues"]
        )

        # 4. Validate external dependencies
        deps_result = self._validate_dependencies()
        results["detailed_results"]["dependencies"] = deps_result
        results["validation_summary"]["dependency_warnings"] = len(
            deps_result["warnings"]
        )

        # 5. Security validation
        security_result = self._validate_security()
        results["detailed_results"]["security"] = security_result
        results["validation_summary"]["security_issues"] = len(
            security_result["issues"]
        )

        # 6. Production readiness assessment
        prod_result = self._assess_production_readiness()
        results["detailed_results"]["production"] = prod_result
        results["validation_summary"]["production_ready"] = prod_result["ready"]

        # Generate overall assessment
        results["validation_summary"][
            "overall_success"
        ] = self._calculate_overall_success(results)
        results["recommendations"] = self._generate_recommendations(results)

        self.validation_results = results
        return results

    def _validate_config_loading(self) -> dict[str, Any]:
        """Validate configuration can be loaded successfully"""
        result = {
            "success": False,
            "error": None,
            "config_type": None,
            "load_time_ms": 0,
        }

        import time

        start_time = time.time()

        try:
            load_validated_config(
                config_file=self.config_file, validate_dependencies=False
            )
            result["success"] = True
            result["config_type"] = "validated"
            logger.info("Configuration loaded successfully using validated system")

        except ConfigurationError as e:
            result["error"] = str(e)
            result["config_type"] = "validation_failed"
            logger.warning(f"Validated configuration failed: {e}")

            # Try legacy configuration
            try:
                LegacyConfig()
                result["config_type"] = "legacy_fallback"
                logger.info("Fell back to legacy configuration system")
            except Exception as legacy_error:
                result[
                    "error"
                ] = f"Both validated and legacy failed. Validated: {e}. Legacy: {legacy_error}"
                result["config_type"] = "failed"

        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            result["config_type"] = "failed"

        result["load_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result

    def _validate_parameters(self) -> dict[str, Any]:
        """Validate individual parameter values"""
        result = {"issues": [], "warnings": [], "parameter_count": 0}

        try:
            config = load_validated_config(
                config_file=self.config_file, validate_dependencies=False
            )

            # Count total parameters
            config_dict = config.model_dump()
            result["parameter_count"] = self._count_parameters(config_dict)

            # All parameters are automatically validated by Pydantic
            # Any issues would have been caught during loading
            result["warnings"].append("All parameters passed Pydantic validation")

        except ConfigurationError as e:
            # Parse Pydantic validation errors for detailed reporting
            if "Configuration validation failed:" in str(e):
                error_lines = str(e).split("\n")[1:]  # Skip header
                for line in error_lines:
                    if line.strip().startswith("•"):
                        result["issues"].append(line.strip()[2:])  # Remove bullet

        return result

    def _validate_constraints(self) -> dict[str, Any]:
        """Validate cross-parameter constraints and logical relationships"""
        result = {"issues": [], "warnings": []}

        try:
            config = load_validated_config(
                config_file=self.config_file, validate_dependencies=False
            )

            # Validate retry configuration logic
            if config.retry.max_delay <= config.retry.initial_delay:
                result["issues"].append(
                    "Retry max_delay must be greater than initial_delay"
                )

            # Validate profiling configuration
            if config.profiling.enabled and config.profiling.sampling_rate == 0.0:
                result["warnings"].append(
                    "Profiling enabled but sampling rate is 0.0 - no data will be collected"
                )

            # Validate server configuration
            if config.server.host == "0.0.0.0" and config.server.log_level == "DEBUG":
                result["warnings"].append(
                    "Server bound to all interfaces with DEBUG logging - potential security risk"
                )

            # Validate session limits
            if config.session.max_active_sessions < 10:
                result["warnings"].append(
                    "Very low max_active_sessions limit may cause session conflicts"
                )

            # Validate model token limits
            if config.model.default_max_tokens > config.model.max_chars_opus // 4:
                result["warnings"].append(
                    "Default max tokens may exceed character limits for some models"
                )

        except ConfigurationError:
            result["issues"].append(
                "Could not load configuration for constraint validation"
            )

        return result

    def _validate_dependencies(self) -> dict[str, Any]:
        """Validate external dependencies and service availability"""
        result = {"warnings": [], "services": {}}

        # Check Python package dependencies
        deps_to_check = ["litellm", "boto3", "httpx", "yaml"]
        for dep in deps_to_check:
            try:
                __import__(dep)
                result["services"][dep] = "available"
            except ImportError:
                result["services"][dep] = "missing"
                result["warnings"].append(f"Optional dependency '{dep}' not installed")

        # Check environment variables for external services
        env_checks = {
            "AWS_PROFILE": "AWS profile for Bedrock access",
            "LITELLM_API_KEY": "LiteLLM API key",
            "CONTEXT_SWITCHER_SECRET_KEY": "Encryption secret key",
        }

        for env_var, description in env_checks.items():
            if os.getenv(env_var):
                result["services"][env_var] = "configured"
            else:
                result["services"][env_var] = "missing"
                if env_var == "CONTEXT_SWITCHER_SECRET_KEY":
                    result["warnings"].append(
                        f"{description} not set - security features limited"
                    )
                else:
                    result["warnings"].append(
                        f"{description} not set - {env_var} backend may not work"
                    )

        return result

    def _validate_security(self) -> dict[str, Any]:
        """Validate security-related configuration"""
        result = {"issues": [], "warnings": [], "security_score": 100}

        try:
            config = load_validated_config(
                config_file=self.config_file, validate_dependencies=False
            )

            # Check secret key configuration
            if not config.security.secret_key:
                result["issues"].append(
                    "No secret key configured - encryption features disabled"
                )
                result["security_score"] -= 30
            elif len(config.security.secret_key) < 32:
                result["issues"].append(
                    "Secret key too short - minimum 32 characters required"
                )
                result["security_score"] -= 20

            # Check server binding
            if config.server.host == "0.0.0.0":
                result["warnings"].append(
                    "Server bound to all interfaces - ensure firewall protection"
                )
                result["security_score"] -= 10

            # Check logging level
            if config.server.log_level == "DEBUG":
                result["warnings"].append(
                    "DEBUG logging enabled - may expose sensitive information"
                )
                result["security_score"] -= 5

            # Check profiling configuration
            if config.profiling.track_costs and config.profiling.level == "detailed":
                result["warnings"].append(
                    "Detailed cost tracking enabled - monitor for sensitive data leakage"
                )
                result["security_score"] -= 5

        except ConfigurationError:
            result["issues"].append(
                "Could not load configuration for security validation"
            )
            result["security_score"] = 0

        return result

    def _assess_production_readiness(self) -> dict[str, Any]:
        """Assess if configuration is suitable for production use"""
        result = {"ready": False, "issues": [], "recommendations": []}

        try:
            config = load_validated_config(
                config_file=self.config_file, validate_dependencies=False
            )

            # Use built-in production readiness check
            result["ready"] = config.is_production_ready

            if not result["ready"]:
                # Identify specific production issues
                if config.server.log_level == "DEBUG":
                    result["issues"].append("DEBUG logging not suitable for production")
                    result["recommendations"].append("Set log_level to INFO or WARNING")

                if not config.security.secret_key:
                    result["issues"].append(
                        "Secret key required for production security"
                    )
                    result["recommendations"].append(
                        "Set CONTEXT_SWITCHER_SECRET_KEY environment variable"
                    )

                if config.profiling.level == "detailed":
                    result["issues"].append(
                        "Detailed profiling may impact production performance"
                    )
                    result["recommendations"].append(
                        "Use 'basic' or 'standard' profiling level"
                    )

                if config.session.cleanup_interval_seconds > 3600:
                    result["issues"].append("Cleanup interval too long for production")
                    result["recommendations"].append(
                        "Set cleanup interval to <= 1 hour"
                    )

        except ConfigurationError:
            result["issues"].append(
                "Could not assess production readiness due to configuration errors"
            )

        return result

    def _calculate_overall_success(self, results: dict[str, Any]) -> bool:
        """Calculate overall validation success"""
        return (
            results["validation_summary"]["loading_success"]
            and results["validation_summary"]["parameter_issues"] == 0
            and results["validation_summary"]["constraint_issues"] == 0
            and results["validation_summary"]["security_issues"] == 0
        )

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []

        if not results["validation_summary"]["loading_success"]:
            recommendations.append("Fix configuration loading errors before proceeding")

        if results["validation_summary"]["parameter_issues"] > 0:
            recommendations.append("Review and fix parameter validation errors")

        if results["validation_summary"]["security_issues"] > 0:
            recommendations.append("Address security configuration issues")

        if not results["validation_summary"]["production_ready"]:
            recommendations.append("Review production readiness recommendations")

        if results["validation_summary"]["dependency_warnings"] > 0:
            recommendations.append(
                "Consider installing optional dependencies for full functionality"
            )

        # Add specific recommendations from production assessment
        prod_recs = (
            results["detailed_results"].get("production", {}).get("recommendations", [])
        )
        recommendations.extend(prod_recs)

        return recommendations

    def _count_parameters(self, config_dict: dict[str, Any], prefix: str = "") -> int:
        """Recursively count configuration parameters"""
        count = 0
        for key, value in config_dict.items():
            if isinstance(value, dict):
                count += self._count_parameters(value, f"{prefix}{key}.")
            else:
                count += 1
        return count

    def _get_timestamp(self) -> str:
        """Get current timestamp for reports"""
        from datetime import datetime

        return datetime.now().isoformat()

    def generate_report(self, format: str = "text") -> str:
        """Generate validation report

        Args:
            format: Report format ('text', 'json', 'markdown')

        Returns:
            Formatted report string
        """
        if not self.validation_results:
            self.validate_all()

        if format == "json":
            return json.dumps(self.validation_results, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate human-readable text report"""
        results = self.validation_results
        lines = [
            "Configuration Validation Report",
            "=" * 50,
            f"Timestamp: {results['timestamp']}",
            f"Config File: {results['config_file'] or 'Environment variables only'}",
            "",
            "VALIDATION SUMMARY",
            "-" * 20,
        ]

        summary = results["validation_summary"]
        lines.extend(
            [
                f"Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}",
                f"Configuration Loading: {'✅ SUCCESS' if summary['loading_success'] else '❌ FAILED'}",
                f"Parameter Issues: {summary['parameter_issues']}",
                f"Constraint Issues: {summary['constraint_issues']}",
                f"Security Issues: {summary['security_issues']}",
                f"Dependency Warnings: {summary['dependency_warnings']}",
                f"Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}",
                "",
            ]
        )

        # Add detailed results
        details = results["detailed_results"]

        # Loading details
        if not summary["loading_success"]:
            lines.extend(
                [
                    "LOADING ERRORS",
                    "-" * 15,
                    f"Error: {details['loading'].get('error', 'Unknown error')}",
                    f"Config Type: {details['loading'].get('config_type', 'unknown')}",
                    "",
                ]
            )

        # Parameter issues
        if summary["parameter_issues"] > 0:
            lines.extend(
                [
                    "PARAMETER ISSUES",
                    "-" * 17,
                ]
            )
            for issue in details["parameters"]["issues"]:
                lines.append(f"• {issue}")
            lines.append("")

        # Constraint issues
        if summary["constraint_issues"] > 0:
            lines.extend(
                [
                    "CONSTRAINT ISSUES",
                    "-" * 17,
                ]
            )
            for issue in details["constraints"]["issues"]:
                lines.append(f"• {issue}")
            lines.append("")

        # Security issues
        if summary["security_issues"] > 0:
            lines.extend(
                [
                    "SECURITY ISSUES",
                    "-" * 16,
                ]
            )
            for issue in details["security"]["issues"]:
                lines.append(f"• {issue}")
            lines.append("")

        # Recommendations
        if results["recommendations"]:
            lines.extend(
                [
                    "RECOMMENDATIONS",
                    "-" * 15,
                ]
            )
            for rec in results["recommendations"]:
                lines.append(f"• {rec}")
            lines.append("")

        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate Markdown-formatted report"""
        results = self.validation_results
        summary = results["validation_summary"]

        md_lines = [
            "# Configuration Validation Report",
            "",
            f"**Timestamp:** {results['timestamp']}  ",
            f"**Config File:** {results['config_file'] or 'Environment variables only'}",
            "",
            "## Validation Summary",
            "",
            f"- **Overall Success:** {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}",
            f"- **Configuration Loading:** {'✅ SUCCESS' if summary['loading_success'] else '❌ FAILED'}",
            f"- **Parameter Issues:** {summary['parameter_issues']}",
            f"- **Constraint Issues:** {summary['constraint_issues']}",
            f"- **Security Issues:** {summary['security_issues']}",
            f"- **Dependency Warnings:** {summary['dependency_warnings']}",
            f"- **Production Ready:** {'✅ YES' if summary['production_ready'] else '❌ NO'}",
            "",
        ]

        # Add detailed sections for any issues
        details = results["detailed_results"]

        if summary["parameter_issues"] > 0:
            md_lines.extend(
                [
                    "## Parameter Issues",
                    "",
                ]
            )
            for issue in details["parameters"]["issues"]:
                md_lines.append(f"- {issue}")
            md_lines.append("")

        if summary["security_issues"] > 0:
            md_lines.extend(
                [
                    "## Security Issues",
                    "",
                ]
            )
            for issue in details["security"]["issues"]:
                md_lines.append(f"- {issue}")
            md_lines.append("")

        if results["recommendations"]:
            md_lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for rec in results["recommendations"]:
                md_lines.append(f"- {rec}")
            md_lines.append("")

        return "\n".join(md_lines)


def validate_config_file(config_file: str, format: str = "text") -> str:
    """Validate a configuration file and return formatted report

    Args:
        config_file: Path to configuration file
        format: Report format ('text', 'json', 'markdown')

    Returns:
        Formatted validation report
    """
    validator = ConfigurationValidator(config_file)
    return validator.generate_report(format)


def validate_current_environment(format: str = "text") -> str:
    """Validate current environment configuration

    Args:
        format: Report format ('text', 'json', 'markdown')

    Returns:
        Formatted validation report
    """
    validator = ConfigurationValidator()
    return validator.generate_report(format)


def check_migration_readiness() -> str:
    """Check if system is ready for configuration migration

    Returns:
        Migration readiness report
    """
    return generate_migration_report()


if __name__ == "__main__":
    """Command-line interface for configuration validation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Context Switcher MCP configuration"
    )
    parser.add_argument("--config", help="Path to configuration file to validate")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--migration",
        action="store_true",
        help="Check migration readiness instead of validating config",
    )
    parser.add_argument("--output", help="Output file (default: stdout)")

    args = parser.parse_args()

    try:
        if args.migration:
            report = check_migration_readiness()
        elif args.config:
            report = validate_config_file(args.config, args.format)
        else:
            report = validate_current_environment(args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            logger.info(f"Configuration validation report written to {args.output}")
            print(
                f"Report written to {args.output}", file=sys.stderr
            )  # MCP protocol compliance: redirect to stderr
        else:
            print(
                report, file=sys.stderr
            )  # MCP protocol compliance: redirect to stderr

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
