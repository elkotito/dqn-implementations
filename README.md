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

Save policy checkpoints:

```bash
uv run python -m dqn.main \
  --config configs/pong.yaml \
  --policy-dir artifacts/pong \
  --policy-save-interval-steps 50000
```

This writes:

- `best_policy.pt`, selected by the highest rolling mean episode return.
- `last_policy.pt`, updated periodically and when training exits.

These files contain policy weights and small inference metadata only. They do not resume training.

Record greedy episodes from a saved policy:

```bash
uv run python -m dqn.record_video artifacts/pong/best_policy.pt \
  --output-dir videos/pong \
  --episodes 3
```

## RunPod training

Create a network volume once:

```bash
runpodctl network-volume create \
  --name dqn-storage \
  --size 10 \
  --data-center-id EUR-NO-1
```

Create a pod once, replacing `NETWORK_VOLUME_ID` with the ID returned above:

```bash
runpodctl pod create \
  --name dqn-training \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --data-center-ids EUR-NO-1 \
  --network-volume-id NETWORK_VOLUME_ID \
  --ports 22/tcp
```

Run training, replacing `POD_ID` with the ID returned above:

```bash
scripts/train.sh POD_ID
```

The script starts the pod, waits for SSH, uploads the current code, runs
training, and stops the pod. Policies remain on the network volume under
`/workspace/dqn/policies`.

runpodctl pod create \
 --name dqn-training \
 --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
 --gpu-id "NVIDIA GeForce RTX 4090" \
 --network-volume-id 8prdhduj4p \
 --ports 22/tcp
