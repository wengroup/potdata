from uuid import uuid4


def suuid() -> str:
    """
    Generate a string UUID (universally unique identifier).

    Uses the UUID4 specification.

    Returns
        A string UUID.
    """
    return str(uuid4())
