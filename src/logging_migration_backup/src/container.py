"""
Dependency Injection Container for Context Switcher MCP

This module provides a simple dependency injection container that enables
loose coupling between modules and supports testability. All dependencies
are registered here and resolved through protocols.
"""

import logging
from typing import TypeVar, Type, Dict, Any, Optional, Callable, cast
from threading import Lock

from .protocols import ContainerProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DependencyResolutionError(Exception):
    """Raised when dependency cannot be resolved"""

    pass


class CircularDependencyError(DependencyResolutionError):
    """Raised when circular dependency is detected"""

    pass


class DependencyContainer(ContainerProtocol):
    """Simple dependency injection container with lifecycle management"""

    def __init__(self):
        self._instances: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._resolving: set = set()  # Track what we're currently resolving
        self._lock = Lock()  # Thread safety for registration

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a pre-created singleton instance

        Args:
            interface: The interface/protocol type
            instance: The instance to register
        """
        with self._lock:
            logger.debug(f"Registering instance for {interface.__name__}")
            self._instances[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function that creates instances

        Args:
            interface: The interface/protocol type
            factory: Function that creates instances
        """
        with self._lock:
            logger.debug(f"Registering factory for {interface.__name__}")
            self._factories[interface] = factory

    def register_singleton_factory(
        self, interface: Type[T], factory: Callable[[], T]
    ) -> None:
        """Register a factory that creates a singleton instance

        The factory will only be called once, and the result will be cached.

        Args:
            interface: The interface/protocol type
            factory: Function that creates the singleton instance
        """
        with self._lock:
            logger.debug(f"Registering singleton factory for {interface.__name__}")

            # Wrap the factory to provide singleton behavior
            def singleton_wrapper():
                if interface not in self._singletons:
                    self._singletons[interface] = factory()
                return self._singletons[interface]

            self._factories[interface] = singleton_wrapper

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient dependency (new instance each time)

        Args:
            interface: The interface/protocol type
            implementation: The concrete implementation class
        """

        def factory():
            return implementation()

        self.register_factory(interface, factory)

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton dependency (same instance each time)

        Args:
            interface: The interface/protocol type
            implementation: The concrete implementation class
        """

        def factory():
            return implementation()

        self.register_singleton_factory(interface, factory)

    def get(self, interface: Type[T]) -> T:
        """Get instance of interface

        Args:
            interface: The interface/protocol type to resolve

        Returns:
            Instance implementing the interface

        Raises:
            DependencyResolutionError: If dependency cannot be resolved
            CircularDependencyError: If circular dependency detected
        """
        # Check for circular dependency
        if interface in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected while resolving {interface.__name__}"
            )

        # Try instances first (pre-registered singletons)
        if interface in self._instances:
            return cast(T, self._instances[interface])

        # Try factories
        if interface in self._factories:
            try:
                self._resolving.add(interface)
                logger.debug(f"Resolving {interface.__name__} via factory")
                instance = self._factories[interface]()
                return cast(T, instance)
            finally:
                self._resolving.discard(interface)

        # No registration found
        raise DependencyResolutionError(
            f"No registration found for {interface.__name__}. "
            f"Available registrations: {list(self._instances.keys()) + list(self._factories.keys())}"
        )

    def get_optional(self, interface: Type[T]) -> Optional[T]:
        """Get instance of interface, return None if not found

        Args:
            interface: The interface/protocol type to resolve

        Returns:
            Instance implementing the interface, or None if not registered
        """
        try:
            return self.get(interface)
        except DependencyResolutionError:
            return None

    def has_registration(self, interface: Type) -> bool:
        """Check if interface is registered

        Args:
            interface: The interface/protocol type to check

        Returns:
            True if interface is registered, False otherwise
        """
        return interface in self._instances or interface in self._factories

    def clear_registration(self, interface: Type) -> bool:
        """Clear registration for interface

        Args:
            interface: The interface/protocol type to clear

        Returns:
            True if registration was found and cleared, False otherwise
        """
        with self._lock:
            cleared = False

            if interface in self._instances:
                del self._instances[interface]
                cleared = True

            if interface in self._factories:
                del self._factories[interface]
                cleared = True

            if interface in self._singletons:
                del self._singletons[interface]
                cleared = True

            if cleared:
                logger.debug(f"Cleared registration for {interface.__name__}")

            return cleared

    def clear_all_registrations(self) -> None:
        """Clear all registrations"""
        with self._lock:
            self._instances.clear()
            self._factories.clear()
            self._singletons.clear()
            logger.debug("Cleared all dependency registrations")

    def get_registration_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about current registrations

        Returns:
            Dictionary with registration information for debugging
        """
        return {
            "instances": {
                interface.__name__: type(instance).__name__
                for interface, instance in self._instances.items()
            },
            "factories": {
                interface.__name__: factory.__name__
                if hasattr(factory, "__name__")
                else "anonymous"
                for interface, factory in self._factories.items()
            },
            "singletons": {
                interface.__name__: type(instance).__name__
                for interface, instance in self._singletons.items()
            },
            "total_registrations": len(self._instances) + len(self._factories),
        }

    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate that all registered dependencies can be resolved

        Returns:
            Validation report with any issues found
        """
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tested_interfaces": [],
        }

        # Test each registration
        for interface in list(self._instances.keys()) + list(self._factories.keys()):
            try:
                # Only test factories, instances are already validated
                if interface in self._factories:
                    self.get(interface)
                    validation_report["tested_interfaces"].append(interface.__name__)
            except CircularDependencyError as e:
                validation_report["valid"] = False
                validation_report["errors"].append(f"Circular dependency: {str(e)}")
            except DependencyResolutionError as e:
                validation_report["valid"] = False
                validation_report["errors"].append(f"Resolution error: {str(e)}")
            except Exception as e:
                validation_report["warnings"].append(
                    f"Unexpected error testing {interface.__name__}: {str(e)}"
                )

        return validation_report


# Global container instance
_global_container: Optional[DependencyContainer] = None
_container_lock = Lock()


def get_container() -> DependencyContainer:
    """Get the global dependency container

    Returns:
        The global DependencyContainer instance
    """
    global _global_container

    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DependencyContainer()
                logger.debug("Created global dependency container")

    return _global_container


def reset_container() -> None:
    """Reset the global container (mainly for testing)"""
    global _global_container

    with _container_lock:
        if _global_container is not None:
            _global_container.clear_all_registrations()
        _global_container = None
        logger.debug("Reset global dependency container")


# Convenience functions that use the global container
def register_instance(interface: Type[T], instance: T) -> None:
    """Register instance in global container"""
    get_container().register_instance(interface, instance)


def register_factory(interface: Type[T], factory: Callable[[], T]) -> None:
    """Register factory in global container"""
    get_container().register_factory(interface, factory)


def register_singleton_factory(interface: Type[T], factory: Callable[[], T]) -> None:
    """Register singleton factory in global container"""
    get_container().register_singleton_factory(interface, factory)


def register_singleton(interface: Type[T], implementation: Type[T]) -> None:
    """Register singleton in global container"""
    get_container().register_singleton(interface, implementation)


def get_dependency(interface: Type[T]) -> T:
    """Get dependency from global container"""
    return get_container().get(interface)


def get_optional_dependency(interface: Type[T]) -> Optional[T]:
    """Get optional dependency from global container"""
    return get_container().get_optional(interface)


def has_dependency(interface: Type) -> bool:
    """Check if dependency is registered in global container"""
    return get_container().has_registration(interface)


# Context manager for temporary dependency overrides (useful for testing)
class DependencyOverride:
    """Context manager for temporarily overriding dependencies"""

    def __init__(self, interface: Type[T], instance: T):
        self.interface = interface
        self.instance = instance
        self.container = get_container()
        self.original_registration = None
        self.had_original = False

    def __enter__(self):
        # Save original registration if it exists
        if self.container.has_registration(self.interface):
            self.had_original = True
            # We can't easily extract the original, so we'll just clear it
            # This is a limitation of the current implementation

        # Register our override
        self.container.register_instance(self.interface, self.instance)
        return self.instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear our override
        self.container.clear_registration(self.interface)

        # Note: We don't restore the original registration in this simple implementation
        # For a production system, you might want to implement a more sophisticated
        # backup/restore mechanism


def with_dependency_override(interface: Type[T], instance: T):
    """Decorator/context manager for dependency override"""
    return DependencyOverride(interface, instance)
