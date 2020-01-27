import os
from typing import Optional, Any

from streamlit.hashing import CodeHasher


def _get_key(obj: Any) -> str:
    hasher = CodeHasher()
    hasher.update(obj)
    return hasher.hexdigest()


class AppCache:
    """Like streamlit's caching, but writes to a local directory."""
    def __init__(self, path):
        self._mem_cache = {}
        self._path = path

    def _get_file_path(self, name: str) -> str:
        return os.path.abspath(f"./{self._path}/{name}")

    def _read_file(self, name: str) -> bytes:
        with open(self._get_file_path(name), "rb") as f:
            return f.read()

    def _write_file(self, name: str, value: bytes) -> None:
        with open(self._get_file_path(name), "wb") as f:
            f.write(value)

    def read(self, obj: Any) -> Optional[bytes]:
        key = _get_key(obj)
        if key not in self._mem_cache:
            try:
                bytes = self._read_file(key)
                self._mem_cache[key] = bytes
            except Exception:
                pass

        return self._mem_cache.get(key, None)

    def write(self, obj: Any, value: bytes) -> None:
        key = _get_key(obj)
        self._write_file(key, value)
        self._mem_cache[key] = value
