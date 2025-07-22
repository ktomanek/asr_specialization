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

# run whisper training locally using the ASRDataset and WhisperTrainer


import json
import shutil
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from asr_dataset import ASRDataset
from whisper_asr_trainer import WhisperTrainer

###
# define dataset 
###

audio_dir = '../../asr_specialization_samples/audio_samples/katrin_german_accent'

# # define dataset and force random splits based on percentage
# my_dataset = ASRDataset(audio_type='mp3', audio_recordings_dir=audio_dir, init_splits_automatically=False)
# my_dataset.init_from_simple_audio_file_dir(test_percentage=0.2, dev_percentage=0.2)

# or automatically initialize splits
my_dataset = ASRDataset(audio_type='mp3', audio_recordings_dir=audio_dir, init_splits_automatically=True)

my_dataset.make_hf_datasets()
my_dataset.show_dataset_stats()
num_train_examples = my_dataset.get_num_train_examples()

###
# define training
###

model_type = 'tiny'
model_size = 'openai/whisper-' + model_type
model_dir = '/tmp/whisper_' + model_type + '_personalized_accent'

# delete if exists
if os.path.exists(model_dir):
    print(f"Deleting existing model dir: {model_dir}")
    shutil.rmtree(model_dir)


eval_batch_size = 4
train_batch_size = 16
train_epochs = 10

# we want to eval ~ 2x per epoch
steps_per_epoch = num_train_examples/train_batch_size
eval_frequency = int(steps_per_epoch / 2)
print(f"steps_per_epoch:{steps_per_epoch}, eval_frequency: {eval_frequency}")

# define trainer
whisper_tiny_trainer = WhisperTrainer(language='en', asr_dataset=my_dataset)
whisper_tiny_trainer.init_trainer(base_model_path_or_name=model_size, 
                                  output_dir=model_dir,
                                  # device settings:
                                  use_mps_device=True, no_cuda=None,
                                  # which part of the model to update:
                                  update_encoder=True, update_decoder=False, update_proj=False,
                                  # trainer settings:
                                  report_to_tensorboard=True,
                                  save_every=eval_frequency, max_saved_checkpoints=2,
                                  max_train_epochs=train_epochs,
                                  train_batch_size=train_batch_size, 
                                  eval_batch_size=eval_batch_size,
                                  eval_on_start=False, eval_every=eval_frequency)

###
# training
###

# run eval before training
res_before = whisper_tiny_trainer.evaluate()
print("Evaluation base model on dev set: ", res_before)
with open(os.path.join(model_dir, 'eval_before_training.txt'), 'w') as f:
    f.write(json.dumps(res_before))

print(">>>> start training")
whisper_tiny_trainer.train()

# save final model
final_model_dir = os.path.join(model_dir, "best_model")
whisper_tiny_trainer.save_model(final_model_dir)

# run eval after training
res_after = whisper_tiny_trainer.evaluate()
print("Evaluation best model on dev set: ", res_after)
with open(os.path.join(model_dir, 'eval_after_training.txt'), 'w') as f:
    f.write(json.dumps(res_after))


