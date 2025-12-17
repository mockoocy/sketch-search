import hashlib


def create_content_hash(image: bytes) -> str:
    """
    Hashes Image content to create a unique identifier.

    Args:
        image: The PIL Image object to be hashed.

    Returns:
        The SHA-256 hash of the image content as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    sha256.update(image)
    return sha256.hexdigest()
