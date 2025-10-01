"""Custom version scheme for setuptools-scm.

This module provides a custom post-release version scheme that omits the
'.post0' suffix when there are zero commits after a tag, while preserving
'.postN' (N > 0) for commits after tags.

Examples:
    - On tag commit: '0.3.34' (not '0.3.34.post0')
    - 5 commits after tag: '0.3.34.post5'
    - Development with uncommitted changes: '0.3.34.post5.dev0+g1234567'
"""

from setuptools_scm.version import get_local_node_and_date


def custom_post_release_scheme(version):
    """Custom version scheme that omits .post0 suffix.
    
    Args:
        version: setuptools_scm version object
        
    Returns:
        str: Formatted version string
    """
    # If this is a tagged version with no commits after (distance == 0 or None)
    if version.distance is None or version.distance == 0:
        # Return just the tag version without .post0
        return version.format_with("{tag}")
    
    # If there are commits after the tag, use post-release format
    return version.format_with("{tag}.post{distance}")


def custom_local_scheme(version):
    """Custom local version scheme.
    
    For development builds (dirty working directory), append .dev0+commit.
    For clean builds, return empty string (no local version).
    
    Args:
        version: setuptools_scm version object
        
    Returns:
        str: Local version string or empty string
    """
    if version.dirty:
        # Development build with uncommitted changes
        return get_local_node_and_date(version)
    
    # Clean build - no local version
    return ""
