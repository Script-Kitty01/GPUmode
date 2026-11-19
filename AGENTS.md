# Environment & Terminal Notes

- **Shell**: The interactive terminal in this workspace uses **`zsh`** (not `bash`).
- **Zsh Syntax & Quirks**:
  - Avoid bash-specific assumptions when running shell commands.
  - The configured `zsh` `preexec` hook emits terminal title OSC escape sequences (`\e\`), which can truncate inline stdout capture in tool responses when running compound `;`-chained commands or inspect commands.
  - When reliable stdout inspection is needed, redirect output to a temporary file and inspect it with `read_files`, or keep commands simple and zsh-compatible.
