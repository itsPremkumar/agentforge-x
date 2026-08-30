# mcp-toolbox

A curated collection of MCP servers with a shared sandbox and protocol helpers.

## Servers

| Server | Entry Point | Description |
|--------|------------|-------------|
| `filesystem-safe` | `mcp-toolbox-filesystem-safe` | Sandboxed filesystem read ops (list, read, stat, find) |
| `git-inspector` | `mcp-toolbox-git-inspector` | Read-only git repository health and metadata tools |
| `board-reader` | `mcp-toolbox-board-reader` | Read-only kanban board query tools (list, get, summary, search) |

## Install

```bash
pip install -e .[dev]
```

## Run

Each server runs over **stdio** transport by default:

```bash
# Install as MCP server in config.yaml or Claude Desktop
mcp-toolbox-filesystem-safe
mcp-toolbox-git-inspector
mcp-toolbox-board-reader
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_TOOLBOX_ROOT` | Sandbox root directory for filesystem-safe and git-inspector | Current working directory |
| `MCP_TOOLBOX_BOARD` | Path to the JSONL board file for board-reader | `<MCP_TOOLBOX_ROOT>/board.jsonl` |

## Client Config Snippet

```json
{
  "mcpServers": {
    "filesystem-safe": {
      "command": "mcp-toolbox-filesystem-safe",
      "env": {
        "MCP_TOOLBOX_ROOT": "/path/to/sandboxed/dir"
      }
    },
    "git-inspector": {
      "command": "mcp-toolbox-git-inspector",
      "env": {
        "MCP_TOOLBOX_ROOT": "/path/to/sandboxed/dir"
      }
    },
    "board-reader": {
      "command": "mcp-toolbox-board-reader",
      "env": {
        "MCP_TOOLBOX_BOARD": "/path/to/board.jsonl"
      }
    }
  }
}
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
