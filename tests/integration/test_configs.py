import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Although `test_train_config` and `test_eval_config` are the same, they can't easily be parametrized because they rely
# on parametrized fixtures. And pytest intentionally doesn't allow the use of the typical `request.getfixturevalue` for
# parametrized fixtures. (see issue here: https://github.com/pytest-dev/pytest/issues/4666)
# It would be possible to use "lazy fixtures" to work around this, but it would require a pytest plugin. At the time of
# writing, it was not considered worth the effort to add a plugin for this one test.


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_eval` pytest fixture.

    Args:
        cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
