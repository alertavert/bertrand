import re
from pathlib import Path
from typing import Optional

from .logger import get_logger


def all_files_in(dirpath: Path,
                 ext: Optional[str] = ".pdf",
                 include: Optional[str] = None,
                 exclude: Optional[str] = None,
                 ) -> list[Path]:
    """List files in a directory tree.

    This function lists all files in a directory tree that match the given criteria; note that
    the matching is **only** based on the file name, not the full directory path.
    If `include` is provided, the `ext` parameter is ignored.

    :param dirpath: The directory dir, can be a simple file dir.
    :param ext: The file extension to filter by, ignored if `include` is present.
    :param include: Include files that match this RegEx pattern.
    :param exclude: Exclude files that match this RegEx pattern. The exclusion pattern is
    applied **after** the inclusion pattern.

    :return: List of absolute file paths that match the criteria, starting from the given
        directory dir. If the dir is a file, the list will contain only that file's absolute dir.

    """
    log = get_logger()
    if not dirpath.exists():
        log.error(f"Path does not exist: {dirpath}")
        return []

    # if ext is provided, we convert it to a regex pattern, if neither
    # include nor exclude is provided.
    if ext:
        if not ext.startswith("."):
            ext = f"\\.{ext}"
        include = include or f".+{ext}$"
    log.info("Patterns: include=%s, exclude=%s", include, exclude)
    # We pre-compile the regex patterns for efficiency
    if include:
        include = re.compile(include)
    if exclude:
        exclude = re.compile(exclude)

    result = []
    if dirpath.is_file():
        if _matches_criteria(dirpath, include, exclude):
            result.append(dirpath.absolute())
    else:
        for curdir, _, files in dirpath.walk():
            for file in files:
                if _matches_criteria(file, include, exclude):
                    result.append(curdir.absolute() / file)
    return result

def _matches_criteria(fname: str,
                      include: Optional[re.Pattern] = None,
                      exclude: Optional[re.Pattern] = None
                      ) -> bool:
    """Check if a file matches the given criteria."""
    log = get_logger()
    if include:
        if not include.match(fname):
            return False
    log.debug("Matches include (%s): %s", include.pattern, fname)
    if exclude:
        if exclude.match(fname):
            log.debug("Excluding (%s): %s", exclude.pattern, fname)
            return False
    return True
