from dqn.training.trainer import Trainer, TrainerConfig


def main() -> None:
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
