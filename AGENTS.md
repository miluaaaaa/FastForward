# Public release conventions

- Use FastForward branding throughout public source, package names, CLI commands, environment variables, configuration examples and documentation.
- Python package: `fastforward_accel`; distribution and CLI: `fastforward-accel`.
- Keep internal workstream identities and private filesystem paths out of published files.
- Preserve model/method attribution and scientific measurement limits.
- Run CPU contract tests and build the wheel after packaging changes. Do not describe these checks as GPU validation.
