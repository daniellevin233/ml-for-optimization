from pathlib import Path


def find_project_root() -> Path:
    """
    Find the project root by walking up directories until finding one with 'instances' folder.

    Returns:
        Path to the project root directory
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / "instances").exists():
            return current
        current = current.parent
    return Path.cwd()