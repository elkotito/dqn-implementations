from training.trainer import Trainer, TrainerConfig


def main():
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
