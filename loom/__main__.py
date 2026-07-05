"""Allow ``python -m loom`` as an alternative to the ``loom`` script."""

import sys

from loom.cli import main

if __name__ == "__main__":
    sys.exit(main())
