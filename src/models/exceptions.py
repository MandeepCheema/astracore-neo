"""
AstraCore Neo Models exceptions.
"""


class ModelsBaseError(Exception):
    """Base exception for all model library errors."""


class ModelRegistrationError(ModelsBaseError):
    """Duplicate or invalid model registration."""


class ModelNotFoundError(ModelsBaseError):
    """Model not found in the catalog."""


class ModelValidationError(ModelsBaseError):
    """Model does not meet hardware or specification constraints."""
