"""Allow ``python -m chiron`` to launch the overlay."""

from __future__ import annotations

from chiron.app import main

if __name__ == "__main__":
    raise SystemExit(main())
