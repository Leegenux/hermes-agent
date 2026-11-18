"""Complete byte inventories for a native capacity skill transaction."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import stat


def direct_path(path: Path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Indirect capacity package path refused')


def package_state(root: Path):
    """Freeze files AND empty directories. No exclusions, execution or UTF-8 coercion."""
    from agent.background_review_capacity import _walk_files, _safe_candidate_name

    direct_path(root)
    if not root.is_dir():
        raise ValueError('Capacity package root is missing')
    files, directories = [], []
    for directory, dirs, names in _walk_files(root):
        for name in dirs:
            directories.append((directory / name).relative_to(root).as_posix())
        for name in names:
            path = directory / name
            direct_path(path)
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise ValueError('Non-regular capacity package member refused')
            digest, count = sha256(), 0
            with path.open('rb') as stream:
                while raw := stream.read(65536):
                    digest.update(raw)
                    count += len(raw)
            files.append((path.relative_to(root).as_posix(), digest.hexdigest(), count))
    paths = [name for name, _, _ in files] + directories
    if (not all(_safe_candidate_name(name) for name in paths)
            or len({name.casefold() for name in paths}) != len(paths)):
        raise ValueError('Unsafe or case-colliding capacity package members')
    return tuple(sorted(files)), tuple(sorted(directories))


def file_manifest(files):
    return {name: {'sha256': digest, 'bytes': count} for name, digest, count in files}
