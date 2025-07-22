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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ondemand'))
import trainer_on_modal
import time

if __name__ == "__main__":

    audio_dir = '../../asr_specialization_samples/audio_samples/katrin_german_accent'
    runid = trainer_on_modal.run_training(
        audio_recordings_dir=audio_dir,
        epochs=10, 
        language='en', 
        whisper_model_type='openai/whisper-small'
        )
    
    t1 = time.time()
    trainer_on_modal.download_best_model_and_save(
        runid=runid,
        local_path='/tmp/modal_training_' + runid)
    t2 = time.time()
    print(f"Total time to download model: {t2-t1}")
    
    trainer_on_modal.delete_run_data(runid) 