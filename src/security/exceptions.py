"""Security module exceptions."""


class SecurityBaseError(Exception):
    """Base exception for all security subsystem errors."""


class SecureBootError(SecurityBaseError):
    """Secure boot chain verification failure."""


class SignatureError(SecurityBaseError):
    """Cryptographic signature verification failure."""


class TEEError(SecurityBaseError):
    """Trusted Execution Environment access violation."""


class OTAError(SecurityBaseError):
    """Over-the-air update error."""


class RollbackError(OTAError):
    """Attempted rollback to an older firmware version."""


class KeyError_(SecurityBaseError):
    """Key management error (named with trailing underscore to avoid shadowing built-in)."""
