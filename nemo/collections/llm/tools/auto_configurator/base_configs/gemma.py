# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from nemo.collections import llm
from nemo.collections.llm.utils import Config

from .basic import Basic


class Gemma(Basic):
    def __init__(
        self,
        name: str = "Gemma",
        version: int = None,
        size: int = 2,
        measure: str = "B",
        cfg: dict = {},
    ):
        """
        :param str name: model name.
        :param int version: model version.
        :param int size: model size.
        :param str measure: meausre of model size. "M" if model size in millions, "B" if in billions.
        :param dict cfg: auto configurator runner config.
        """

        super().__init__(name=name, version=version, size=size, measure=measure, cfg=cfg)
        self.config_name = f"{self.name}Config{self.size}{self.measure}"

    def get_model_config(self) -> Config:
        """
        Function that returns model config.
        :return: model config.
        :rtype: Config.
        """

        model_class = getattr(llm, self.config_name)
        kwargs = self.cfg.get("model_args", {})

        if self.nemo_run:
            model_config = Config(model_class, **kwargs)
        else:
            model_config = model_class(**kwargs)

        model_config.global_batch_size = self.global_batch_size
        model_config.seq_length = self.seq_length
        model_config.pipeline_dtype = torch.bfloat16

        return model_config
