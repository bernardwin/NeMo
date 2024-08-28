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

from nemo.collections.llm.tools.auto_configurator import base_configs

from .basic import Basic


def custom(name, cfg):
    """
    Function taht return custom model class.
    :param dict cfg: auto configurator runner config.
    :return: Custom class object.
    """
    basic_class = getattr(base_configs, name)

    class Custom(basic_class):
        def __init__(self, name, cfg):
            """
            :param str name: model name.
            :param dict cfg: auto configurator runner config.
            """

            super().__init__(name=name, cfg=cfg)

    custom_class = Custom(name, cfg)

    return custom_class
