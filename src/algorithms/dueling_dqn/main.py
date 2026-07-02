from src.algorithms.dueling_dqn.config import DuelingDQNConfig
from src.algorithms.dueling_dqn.trainer import DuelingDQNTrainer


def main() -> None:
    config = DuelingDQNConfig()
    trainer = DuelingDQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
