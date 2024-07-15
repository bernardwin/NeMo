import os

from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig

from .basic import Basic


class Llama(Basic):
    def __init__(
        self,
        name: str = "Llama",
        version: int = 2,
        size: int = 7,
        cfg: dict = {},
    ):
        super().__init__(name=name, version=version, size=size, cfg=cfg)
        self.config_name = f"{self.name}{self.version}Config{self.size}B"

    def get_model_config(self):
        model_config = getattr(llm, self.config_name)
        model_config.global_batch_size = self.global_batch_size
        model_config.activations_checkpoint_method = None
        model_config.seq_length = self.seq_length

        return model_config

    def get_optim_config(self):
        optim_config = OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            min_lr=1e-5,
            use_distributed_optimizer=True,
            bf16=True,
            adam_beta1=0.9,
            adam_beta2=0.95,
            overlap_grad_reduce=False,
            overlap_param_gather=True,
        )

        return optim_config

    def get_tokenizer_config(self):
        tokenizer_config = {
            "library": "sentencepiece",
            "tokenizer_model": None,
            "legacy": False,
            "chat_template": None,
        }

        return tokenizer_config

    def get_trainer_config(self):
        trainer_config = {
            "accelerator": "gpu",
            "precision": "bf16",
            "logger": False,
            "enable_checkpointing": False,
            "use_distributed_sampler": False,
            "max_epochs": None,
            "log_every_n_steps": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "num_nodes": self.num_nodes,
            "devices": self.num_gpus,
            "max_steps": self.max_steps,
        }

        return trainer_config

    def get_data_config(self):
        data_config = {
            "paths": None,
            "weights": None,
            "seq_length": self.seq_length,
            "global_batch_size": self.global_batch_size,
            "num_workers": 2,
            "split": "99990,8,2",
        }

        return data_config

    def get_run_config(self):
        run_config = {
            "name": f"llama{self.version}_{self.size}b",
            "results_dir": None,
            "time_limit": "0-00:30:00",
            "dependency": "singleton",
        }

        return run_config
