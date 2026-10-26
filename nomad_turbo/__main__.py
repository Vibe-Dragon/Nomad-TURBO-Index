"""Command entry. `python3 -m nomad_turbo` calls cli.main.

This does not patch the NOMAD admin container, and importing it does not
start indexing.
"""

from nomad_turbo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
