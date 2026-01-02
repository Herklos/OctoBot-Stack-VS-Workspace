---
description: "Guidelines for creating and updating VS Code launch.json and tasks.json configurations for debugging and task automation"
globs: [".vscode/launch.json", ".vscode/tasks.json"]
alwaysApply: false
---

# VS Code Launch and Task Configuration Rules

## File Structure

VS Code configuration files are located in `.vscode/` directory:
- `launch.json` - Debug configurations
- `tasks.json` - Task definitions

Both files use JSON format with comments support (via `//` or `/* */`).

## launch.json Structure

### Required Properties

Every `launch.json` must have:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "debugpy",  // or "node", "cppdbg", etc.
      "request": "launch",  // or "attach"
      "name": "Descriptive Name"
    }
  ]
}
```

### Common Configuration Properties

When creating Python debug configurations (OctoBot project):

**Required:**
- `type`: `"debugpy"` for Python debugging
- `request`: `"launch"` to start program, `"attach"` to attach to running process
- `name`: Unique, descriptive name shown in debug dropdown

**Common:**
- `program`: Entry point file (e.g., `"${workspaceFolder}/OctoBot/start.py"`)
- `module`: Python module to run (e.g., `"pytest"` for tests)
- `cwd`: Working directory (e.g., `"${workspaceFolder}/OctoBot"`)
- `console`: `"integratedTerminal"` or `"internalConsole"` or `"externalTerminal"`
- `args`: Array of command-line arguments
- `env`: Environment variables object
- `preLaunchTask`: Name of task to run before debugging (must match task label)

**Python-specific:**
- `justMyCode`: `false` to debug into library code
- `python`: Path to Python interpreter (optional, uses default if omitted)

**Presentation (optional):**
- `presentation.hidden`: `false` to show in debug dropdown
- `presentation.group`: Group name for organizing configurations
- `presentation.order`: Numeric order within group

### Variable Substitution

Use VS Code variables in paths and values:
- `${workspaceFolder}` - Root workspace folder
- `${file}` - Currently active file
- `${env:VARIABLE_NAME}` - Environment variable
- `${command:commandId}` - VS Code command result

**Example:**
```json
{
  "program": "${workspaceFolder}/OctoBot/start.py",
  "cwd": "${workspaceFolder}/OctoBot",
  "args": ["${env:USERNAME}"]
}
```

### Platform-Specific Properties

Override properties per platform:
```json
{
  "args": ["default-arg"],
  "windows": {
    "args": ["windows-arg"]
  },
  "linux": {
    "args": ["linux-arg"]
  },
  "osx": {
    "args": ["macos-arg"]
  }
}
```

**Note:** `type` cannot be in platform-specific section.

### Compound Launch Configurations

Launch multiple debug sessions simultaneously:
```json
{
  "version": "0.2.0",
  "configurations": [
    { "name": "Server", "type": "node", "request": "launch", "program": "server.js" },
    { "name": "Client", "type": "node", "request": "launch", "program": "client.js" }
  ],
  "compounds": [
    {
      "name": "Server/Client",
      "configurations": ["Server", "Client"],
      "preLaunchTask": "${defaultBuildTask}",
      "stopAll": true
    }
  ]
}
```

### Server Ready Action

Automatically open browser when server starts:
```json
{
  "serverReadyAction": {
    "pattern": "listening on port ([0-9]+)",
    "uriFormat": "http://localhost:%s",
    "action": "openExternally"
  }
}
```

Actions: `"openExternally"`, `"debugWithEdge"`, `"debugWithChrome"`, or `"startDebugging"` with `"name"` property.

## tasks.json Structure

### Required Properties

Every `tasks.json` must have:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Task Name",
      "type": "shell",  // or "process"
    }
  ]
}
```

### Common Task Properties

**Required:**
- `label`: Unique name (used in `preLaunchTask` references)
- `type`: `"shell"` for shell commands, `"process"` for single executable

**Common:**
- `command`: Command to execute (e.g., `"npm"`, `"python"`, or full path)
- `args`: Array of command arguments
- `options.cwd`: Working directory
- `options.env`: Environment variables object
- `dependsOn`: Array of task labels that must complete first
- `dependsOrder`: `"parallel"` or `"sequence"` for dependencies

**Presentation:**
- `presentation.reveal`: `"always"`, `"silent"`, or `"never"`
- `presentation.panel`: `"shared"`, `"dedicated"`, or `"new"`
- `presentation.focus`: `true` to focus terminal when task runs
- `presentation.clear`: `true` to clear terminal before running

**Problem Matchers:**
- `problemMatcher`: Array of problem matcher names or objects
- Use `"$tsc"` for TypeScript, `"$eslint-stylish"` for ESLint, etc.
- Empty array `[]` for tasks that don't produce parseable output

### Task Dependencies

Chain tasks with `dependsOn`:
```json
{
  "label": "Build",
  "dependsOn": ["Setup Environment", "Install Dependencies"],
  "dependsOrder": "sequence"
}
```

### OctoBot-Specific Patterns

When creating tasks for OctoBot Stack:

**PYTHONPATH Setup:**
```json
{
  "label": "Setup PYTHONPATH",
  "type": "shell",
  "command": "echo 'PYTHONPATH configuration defined in task options.env'",
  "options": {
    "env": {
      "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/Async-Channel:${workspaceFolder}/OctoBot-Commons:..."
    }
  },
  "problemMatcher": [],
  "presentation": {
    "reveal": "silent",
    "panel": "shared"
  }
}
```

**CCXT Build Tasks:**
```json
{
  "label": "CCXT: Build polymarket exchange python",
  "type": "shell",
  "command": "npm run emitAPI polymarket && npm run transpileRest polymarket",
  "options": {
    "cwd": "${workspaceFolder}/ccxt"
  },
  "problemMatcher": [],
  "presentation": {
    "reveal": "silent",
    "panel": "shared"
  }
}
```

**Tentacle Generation:**
```json
{
  "label": "Generate tentacles from OctoBot-Tentacles",
  "type": "shell",
  "command": "${command:python.interpreterPath} start.py tentacles -p ../../tentacles_default_export.zip -d ../OctoBot-Tentacles",
  "dependsOn": ["Setup PYTHONPATH", "CCXT: Build polymarket exchange python"],
  "options": {
    "cwd": "${workspaceFolder}/OctoBot",
    "env": {
      "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/Async-Channel:..."
    }
  },
  "problemMatcher": [],
  "presentation": {
    "reveal": "always",
    "panel": "shared"
  }
}
```

## Best Practices

### Naming Conventions

- Use descriptive, action-oriented names: `"Start OctoBot"`, `"CCXT: Build polymarket exchange python"`
- Prefix related tasks: `"CCXT: ..."` for CCXT-related tasks
- Use groups in launch configs: `"1.Run"`, `"2.Test"`, `"OctoBot-Tentacles-Manager"`

### Organization

**Launch Configurations:**
- Group related configs using `presentation.group`
- Order with `presentation.order` (lower numbers first)
- Use `presentation.hidden: false` to keep configs visible

**Tasks:**
- Keep setup tasks (like PYTHONPATH) with `reveal: "silent"`
- Use `panel: "shared"` to reuse terminal
- Chain related tasks with `dependsOn`

### Environment Variables

- Use `options.env` in tasks to set environment variables
- Use `${env:VARIABLE_NAME}` to reference existing environment variables
- Set PYTHONPATH in tasks that need it, not globally

### Error Handling

- Use appropriate `problemMatcher` for tasks that produce parseable output
- Use empty array `[]` for informational tasks (like PYTHONPATH setup)
- Ensure `preLaunchTask` names match task `label` exactly

### Debugging Configuration

- Set `justMyCode: false` when debugging into library code (OctoBot, CCXT)
- Use `console: "integratedTerminal"` for Python scripts to see output
- Use `module: "pytest"` for test configurations instead of `program`

## Common Patterns

### Python Script Launch
```json
{
  "type": "debugpy",
  "request": "launch",
  "name": "Run Script",
  "program": "${workspaceFolder}/script.py",
  "console": "integratedTerminal",
  "cwd": "${workspaceFolder}",
  "justMyCode": false
}
```

### Pytest Test Launch
```json
{
  "type": "debugpy",
  "request": "launch",
  "name": "Run Tests",
  "module": "pytest",
  "console": "integratedTerminal",
  "cwd": "${workspaceFolder}",
  "args": [
    "tests",
    "-v",
    "-k", "test_name"
  ],
  "justMyCode": false
}
```

### Task with Pre-requisites
```json
{
  "label": "Build and Test",
  "type": "shell",
  "command": "npm test",
  "dependsOn": ["Install Dependencies", "Build"],
  "dependsOrder": "sequence",
  "problemMatcher": []
}
```

## Validation Checklist

Before finalizing launch.json or tasks.json:

- [ ] JSON syntax is valid (use JSON validator)
- [ ] All `preLaunchTask` values match existing task `label` values
- [ ] All `dependsOn` task labels exist
- [ ] Variable substitutions use correct syntax (`${workspaceFolder}`, etc.)
- [ ] File paths use forward slashes or `${workspaceFolder}` variables
- [ ] Task labels are unique
- [ ] Launch configuration names are unique
- [ ] Environment variables are properly scoped (task-level vs global)
- [ ] `problemMatcher` is appropriate (empty array `[]` for non-parseable output)

## References

- [VS Code Debug Configuration](https://code.visualstudio.com/docs/debugtest/debugging-configuration)
- [VS Code Tasks](https://code.visualstudio.com/docs/debugtest/tasks)
- [Variable Substitution](https://code.visualstudio.com/docs/editor/variables-reference)

