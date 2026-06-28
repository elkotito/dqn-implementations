# DQN

Run with the default configuration:

```bash
uv run python -m dqn.algorithms.replay_dqn.main
```

Load configuration from YAML:

```bash
uv run python -m dqn.algorithms.replay_dqn.main --config configs/pong.yaml
```

CLI arguments and `DQN_*` environment variables can override values from the
file:

```bash
DQN_DEVICE=mps uv run python -m dqn.algorithms.replay_dqn.main \
  --config configs/pong.yaml \
  --learning-rate 0.0005
```

Configuration priority is CLI, environment variables, YAML, then defaults. See all available arguments with:

```bash
uv run python -m dqn.algorithms.replay_dqn.main --help
```

Disable Weights & Biases for a dry run:

```bash
uv run python -m dqn.algorithms.replay_dqn.main \
  --config configs/pong.yaml \
  --wandb-mode disabled
```

Policy checkpoints are saved to `./policies` by default. To use a different directory:

```bash
uv run python -m dqn.algorithms.replay_dqn.main \
  --config configs/pong.yaml \
  --policy-dir artifacts/pong
```

This writes:

- `policy.pt`, updated after every periodic greedy evaluation.
- `best_policy.pt`, selected by the highest mean return from periodic greedy evaluation.

These files contain policy weights and small inference metadata only. They do not resume training.

By default, training metrics are logged under `train/*`. Every 250,000 steps,
the policy is evaluated greedily (`epsilon=0`) for 10 episodes in a separate
environment and the results are logged under `eval/*`. Configure this with
`eval_interval_steps` and `eval_episodes`.

Record greedy episodes from a saved policy:

```bash
uv run python -m dqn.record_video artifacts/pong/best_policy.pt \
  --output-dir videos/pong \
  --episodes 3
```

## RunPod training

Create a `WANDB_API_KEY` secret in the
[RunPod console](https://console.runpod.io/user/secrets). RunPod secrets cannot
currently be created with `runpodctl`.

Create a network volume once. It keeps policy checkpoints after the pod stops:

```bash
runpodctl network-volume create \
  --name dqn-storage \
  --size 10 \
  --data-center-id EUR-NO-1
```

List volumes if you need to retrieve its ID:

```bash
runpodctl network-volume list
```

Create a pod once, replacing `NETWORK_VOLUME_ID` with the volume ID. The
environment variable references the secret without exposing its value:

```bash
runpodctl pod create \
  --name dqn-training \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --data-center-ids EUR-NO-1 \
  --network-volume-id NETWORK_VOLUME_ID \
  --ports 22/tcp \
  --env '{"WANDB_API_KEY":"{{ RUNPOD_SECRET_WANDB_API_KEY }}"}'
```

To attach the secret to an existing pod:

```bash
runpodctl pod update POD_ID \
  --env '{"WANDB_API_KEY":"{{ RUNPOD_SECRET_WANDB_API_KEY }}"}'
```

Updating environment variables restarts the pod. Include any other custom
environment variables in the JSON object because the update may replace the
existing environment map.

Run training, replacing `POD_ID` with the ID returned above:

```bash
scripts/train.sh POD_ID
```

The script starts the pod, waits for SSH, uploads the current code, runs
training, and stops the pod. Policies remain on the network volume under
`/workspace/dqn/policies`.

# TODO:

1. Parallel envs
2. Preallocate tensors for transitions
3. Store a single frame and then expand since it is almost exactly the same.
