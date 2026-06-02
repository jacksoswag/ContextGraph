from __future__ import annotations

import subprocess, sys


# Install the English Coreferee model into the active Python environment.
def main() -> int:
    subprocess.check_call([sys.executable, "-m", "coreferee", "install", "en"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
