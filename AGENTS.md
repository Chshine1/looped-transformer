# Project guidance

- Use Pixi for Python commands; bare `python` resolves to a Windows Store alias.
- For CLUTRR generation and preparation, use `pixi run -e clutrr-data python ...`.
- For training and general project work, use `pixi run -e default ...`.
- Run checks appropriate to the change: `pixi run test` or `pixi run typecheck`.
- Normal Pixi execution works in the sandbox. Use `--as-is` only when deliberately using the installed environment without updating dependencies.
- If a command fails because of sandbox permissions, request narrowly scoped escalation instead of stopping or changing system permissions.
