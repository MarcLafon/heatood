import os


def expand_path(pth: str) -> str:
    pth = os.path.expandvars(pth)
    pth = os.path.expanduser(pth)
    return pth
