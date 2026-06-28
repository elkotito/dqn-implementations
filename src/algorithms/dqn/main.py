from src.algorithms.dqn.config import DQNConfig
from src.algorithms.dqn.trainer import DQNTrainer


def main() -> None:
    config = DQNConfig()
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
