from typing import Optional

import nemo_run as run
import pytorch_lightning as pl
import torch

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.recipes import mixtral_8x3b

NAME = "mixtral_8x3b_64k"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    model_config = mixtral_8x3b.model()
    model_config.config.seq_length = 65536
    return model_config


def trainer(
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Config:
    return mixtral_8x3b.trainer(
        tensor_parallelism=4,
        pipeline_parallelism=4,
        pipeline_parallelism_type=torch.bfloat16,
        virtual_pipeline_parallelism=8,
        context_parallelism=4,
        sequence_parallelism=True,
        expert_parallelism=4,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = mixtral_8x3b.pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = mixtral_8x3b.finetune_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    recipe.model = model()
    recipe.trainer = trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)

    return recipe
