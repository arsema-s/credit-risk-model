"""
API entry point.
"""


def health_check():
    """
    Basic health check.
    """
    return {
        "status": "healthy"
    }