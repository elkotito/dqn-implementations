# DQN

Run with the default configuration:

```bash
uv run python -m dqn.main
```

Load configuration from YAML:

```bash
uv run python -m dqn.main --config configs/pong.yaml
```

CLI arguments and `DQN_*` environment variables can override values from the
file:

```bash
DQN_DEVICE=mps uv run python -m dqn.main \
  --config configs/pong.yaml \
  --learning-rate 0.0005
```

Configuration priority is CLI, environment variables, YAML, then defaults. See all available arguments with:

```bash
uv run python -m dqn.main --help
```
