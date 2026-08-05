"""
Operational CLIs
================
Commands an operator runs, not code the API imports.

    edupilot-reindex   index status, blue/green rebuild, promote, rollback
    edupilot-evaluate  run the 50-case suite against the local pipeline

Both are also runnable as modules — `python -m edupilot.cli.reindex --status`.
Each command's `main()` returns a process exit code rather than calling
`sys.exit`, so it stays callable from a test.
"""

__all__: list[str] = []
