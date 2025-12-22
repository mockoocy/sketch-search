import hashlib

from PIL.Image import Image


def create_content_hash(image: Image) -> str:
    """
    Hashes Image content to create a unique identifier.

    Args:
        image: The PIL Image object to be hashed (RGB).

    Returns:
        The SHA-256 hash of the image content as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    sha256.update(image.tobytes())
    return sha256.hexdigest()
