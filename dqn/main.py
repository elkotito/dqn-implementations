from dqn.loggers.wandb import WandbLogger
from dqn.training.trainer import Trainer, TrainerConfig


def main() -> None:
    config = TrainerConfig()
    logger = WandbLogger(project="pong", config=config.model_dump())
    trainer = Trainer(config, logger)
    trainer.train()


if __name__ == "__main__":
    main()
