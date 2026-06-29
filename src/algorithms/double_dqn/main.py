from src.algorithms.double_dqn.config import DoubleDQNConfig
from src.algorithms.double_dqn.trainer import DoubleDQNTrainer


def main() -> None:
    config = DoubleDQNConfig()
    trainer = DoubleDQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
