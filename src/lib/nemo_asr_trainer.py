# Copyright 2025 Katrin Tomanek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from asr_trainer import ASRTrainer
from asr_dataset import ASRDataset

# TODO add training code for NVIDIA Nemo models here as well (fastconformer CTC and RNNT)

class NemoTrainer(ASRTrainer):
    def __init__(self, language, asr_dataset: ASRDataset):
        raise NotImplementedError("init not implemented")

    def init_trainer(self, base_model_path_or_name, output_dir, **kwargs):
        raise NotImplementedError("init not implemented")
    
    def train(self):
        raise NotImplementedError("train not implemented")

    def evaluate(self):
        raise NotImplementedError("evaluate not implemented")
    
    def save_model(self, output_dir):
        raise NotImplementedError("save_model not implemented")    