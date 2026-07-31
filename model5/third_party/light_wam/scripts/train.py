import hydra
from omegaconf import DictConfig

from model5.third_party.light_wam.src.lightwam.runtime import run_training
from model5.third_party.light_wam.src.lightwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
