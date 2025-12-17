class ImageFormatError(ValueError):
    """Raised when an image has an invalid format."""


class InvalidImageError(ValueError):
    """Raised when an image is invalid or corrupted."""


class InvalidFsAccessError(PermissionError):
    """Raised when there is an invalid filesystem access attempt."""


class ImageNotFoundError(LookupError):
    """Raised when an image is not found in the repository."""
