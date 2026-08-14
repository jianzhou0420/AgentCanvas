from __future__ import annotations
class OAuthFlowError(Exception):
    """Base exception for OAuth flow errors."""


class OAuthTokenError(OAuthFlowError):
    """Raised when token operations fail."""


class OAuthRegistrationError(OAuthFlowError):
    """Raised when client registration fails."""
