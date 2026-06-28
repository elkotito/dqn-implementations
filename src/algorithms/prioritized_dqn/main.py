from src.algorithms.prioritized_dqn.config import PrioritizedDQNConfig
from src.algorithms.prioritized_dqn.trainer import PrioritizedDQNTrainer


def main() -> None:
    config = PrioritizedDQNConfig()
    trainer = PrioritizedDQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
