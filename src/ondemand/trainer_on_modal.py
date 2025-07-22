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

# Run training via Modal on GPU in an on-demand environment.
#
# You need to have Modal installed on your machine (see 'getting started' under https://modal.com/docs/guide).

import logging
import modal
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time 
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),
    ]
)

# Note: the way Modal >= 1.0.0 needs local modules to be imported is 
# both unintuitive and also very poorly described on their website.
# What needs to be done is:
# - add the directory to sys.path (or make it otherwise available)
# - add the specific module with add_local_python_source() -- as a string!
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))


logging.info("Starting Modal app...")
TRAINING_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg", "libsndfile1-dev"])
    .pip_install(
        "torch", #tested: 2.7.1",
        "torchcodec", #tested: 0.4.0",
        "numpy", #tested: 2.3.1",
        "accelerate", #tested: 1.9.0",
        "huggingface_hub[hf_transfer]", #tested: 0.33.4",
        "transformers", #tested: 4.53.2",
        "datasets", #tested: 4.0.0",
        "tensorboard",
        "soundfile",
        "evaluate",
        "jiwer",
    )
.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
.add_local_python_source("asr_dataset")
.add_local_python_source("asr_trainer")
.add_local_python_source("whisper_asr_trainer")
)

VOLUME = modal.Volume.from_name("asr-tuning-runs", create_if_missing=True)
VOLUME_MOUNT_PATH = Path('/data')
app = modal.App(name='on-demand-asr-tuning',
                volumes={VOLUME_MOUNT_PATH: VOLUME},
                )
logging.info(f"Modal app name: {app.name}")
logging.info("Modal app successfully started!")

def _create_tgz_archive(directory_path, archive_name):
    """
    Creates a .tgz archive from a given directory (exclude the directory itself and only include its contents).
    This should be the audio data directory, containing the audio recordings (wav/mp3) and prompt files (txt).

    Args:
        directory_path (str): The path to the directory containing the content to archive.
        archive_name (str): The desired name of the archive (e.g., "my_archive.tgz").
    """
    try:
        with tarfile.open(archive_name, "w:gz") as tar:
            # Add only the contents of the directory, not the directory itself
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                tar.add(file_path, arcname=file_name)  # Use the file name as the archive name
        logging.info(f"Archive '{archive_name}' created successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Directory '{directory_path}' not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def _create_run_id():
    """
    Creates a unique run ID based on the current timestamp and a UUID.
    This can be used to uniquely identify a training run and associated data and logs.
    """
    return str(uuid.uuid4())

def _get_run_base_dir(runid):
    run_base_dir = os.path.join(VOLUME_MOUNT_PATH, runid)
    return run_base_dir

def _get_log_dir(runid):
    log_dir = os.path.join(_get_run_base_dir(runid), 'training')
    return log_dir

def _get_best_model_dir(runid):
    best_model_dir = os.path.join(_get_log_dir(runid), 'best_model')
    return best_model_dir

def _upload_data(runid, modal_volume, filename):
    """Uploads the packaged data to the given Modal volume."""
    with modal_volume.batch_upload() as batch:
        tmp_filename = os.path.join(runid, 'audio_dataset', 'data.tar.gz')
        batch.put_file(filename, '/' + tmp_filename)

        logging.info(f"Uploaded data successfully on volume to: {tmp_filename}.")
        return tmp_filename

@app.function()
def _unpack_data(tgz_filename):
    """Unpack data on a Modal volume."""
    
    basename = os.path.dirname(tgz_filename)
    target_dir = os.path.join(VOLUME_MOUNT_PATH, basename)   
    tgz_path = os.path.join(VOLUME_MOUNT_PATH, tgz_filename)
    logging.info(f"unpacking data from: {tgz_path}")
    logging.info(f"unpacking into: {target_dir}")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        members = tar.getmembers()
        tar.extractall(target_dir, members=members)
    os.remove(tgz_path)

    # update the volumne to make changes visible
    VOLUME.commit()
    return target_dir


# time out on purpose low (30 mins) as this is often enough for small personalization scenarios; but should
# be increased for larger datasets or more epochs or larger models.
# CPU is mainly important for feature extraction
@app.function(image=TRAINING_IMAGE,
              gpu='A100',
              cpu=12.0, 
              timeout=3600,
              )
def _train_model(audio_recordings_dir, 
                 log_dir,
                 whisper_model_type, 
                 language,
                 epochs=10):

    import json

    from asr_dataset import ASRDataset
    from whisper_asr_trainer import WhisperTrainer

    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Training on Whisper model type: {whisper_model_type}, for {epochs} epochs.")
    logging.info(f"Writing training logs to: {log_dir}")

    # load dataset (automatically initialize by either using existing metadata files or randomly splitting based on default proportions)
    my_dataset = ASRDataset(audio_type='mp3', audio_recordings_dir=audio_recordings_dir, init_splits_automatically=True)
    my_dataset.make_hf_datasets()
    my_dataset.show_dataset_stats()
    num_train_examples = my_dataset.get_num_train_examples()
    VOLUME.commit()

    eval_batch_size = 8
    train_batch_size = 16

    # we want to eval ~ 2x per epoch
    steps_per_epoch = num_train_examples / train_batch_size
    eval_frequency = int(steps_per_epoch / 2)
    print(f"steps_per_epoch:{steps_per_epoch}, eval_frequency: {eval_frequency}")

    # save checkpoint every time we evaluate (since we're running very few epochs anyways)
    # for longer training runs this may be too much
    save_frequency = eval_frequency

    # load trainer
    whisper_trainer = WhisperTrainer(language=language, asr_dataset=my_dataset)
    whisper_trainer.init_trainer(
        base_model_path_or_name=whisper_model_type, 
        output_dir=log_dir,
        # device settings:
        no_cuda=False,
        # which part of the model to update:
        update_encoder=True, update_decoder=False, update_proj=False,
        # trainer settings:
        report_to_tensorboard=True,
        save_every=save_frequency, max_saved_checkpoints=2,
        max_train_epochs=epochs,
        train_batch_size=train_batch_size, 
        eval_batch_size=eval_batch_size,
        eval_on_start=True, eval_every=eval_frequency)

    # run eval before training
    base_model_perf = whisper_trainer.evaluate()
    print("Evaluation base model on dev set: ", base_model_perf)    
    with open(os.path.join(log_dir, 'eval_before_training.txt'), 'w') as f:
        f.write(json.dumps(base_model_perf))

    print(">>>> start training")
    whisper_trainer.train()

    # save best model
    logging.info(f"Saving the best model after training.")
    best_model_dir = os.path.join(log_dir, 'best_model')
    whisper_trainer.save_model(best_model_dir)
    logging.info(f"Best model saved to: {best_model_dir}")
    VOLUME.commit()

    # run final eval
    final_model_perf = whisper_trainer.evaluate()
    print("Evaluation best model on dev set: ", final_model_perf)
    with open(os.path.join(log_dir, 'eval_after_training.txt'), 'w') as f:
        f.write(json.dumps(final_model_perf))

    return {'base model performance': base_model_perf,
            'final model performance': final_model_perf,
            'best_model_dir': best_model_dir}


@app.function()
def _cleanup(runid):
    """This will delete all the data of the run; make sure to download the model first if you want to keep it!"""
    import shutil    
    run_dir = _get_run_base_dir(runid)
    try:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)  # Recursively delete the directory and its contents
            logging.info(f"Successfully deleted: {run_dir}")
        else:
            logging.error(f"Directory not found: {run_dir}")
    except Exception as e:
        logging.error(f"An error occurred while deleting data: {e}")

# for larger models, might need to allow more memory
@app.function(image=TRAINING_IMAGE, memory=4096, timeout=180)
def _download_best_model(runid):
    from transformers import WhisperForConditionalGeneration
    import time
    best_model_dir = _get_best_model_dir(runid)
    logging.info(f"Obtaining best model from: {best_model_dir}")
    import os
    t1 = time.time()
    best_model = WhisperForConditionalGeneration.from_pretrained(best_model_dir, local_files_only=True)
    t2 = time.time()
    print(f"Time to load model: {t2-t1}")

    print("Returning model to caller...")
    return best_model
    
def download_best_model_and_save(runid, local_path):
    """Downloads the best model from the Modal volume to the local path."""
    with modal.enable_output():
        with app.run():
            best_model = _download_best_model.remote(runid)
            print(f">> obtained model; number of parameters:", str(best_model.num_parameters()))
            best_model.save_pretrained(local_path, safe_serialization=False)
            print(f">> saved model to {local_path}")

def delete_run_data(runid):
    """Will delete all data stored in the Modal volume for the given runid.
    Make sure to download the model first if you want to keep it!"""
    with modal.enable_output():
        with app.run():
            _cleanup.remote(runid)
            print(f">> deleted run data for run {runid}")

def run_training(audio_recordings_dir, whisper_model_type='openai/whisper-tiny', language='en', epochs=2):
    """Run the training on Modal.

    Args:
        audio_recordings_dir: the directory containing the audio recordings
        whisper_model_type: whisper model name to download from huggingface
        language: language code, must be supported by the Whisper models
        epochs: number of training epochs
    Returns:
        runid: the run ID for this training run
    """

    # package directory and upload to Modal volume
    runid = _create_run_id()
    print('>>> run ID:', runid)
    
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', dir='/tmp', delete=True) as temp_f:
        tarfile = temp_f.name
        _create_tgz_archive(audio_recordings_dir, tarfile)
        print('>>> created tgz archive:', tarfile, ' --> from:', audio_recordings_dir)
        remote_tarfile = _upload_data(runid, VOLUME, tarfile)

    # using this will forward all log outputs from remotely run functions here
    with modal.enable_output():
        with app.run():

            remote_audio_recordings_dir = _unpack_data.remote(remote_tarfile)
            print('>>> remote audio_data dir:', remote_audio_recordings_dir)
            
            log_dir = _get_log_dir(runid)
            print('>>> running training on:', log_dir)
            
            t1 = time.time()
            result = _train_model.remote(
                audio_recordings_dir=remote_audio_recordings_dir, 
                log_dir=log_dir,
                language=language,
                whisper_model_type=whisper_model_type, epochs=epochs)
            t2 = time.time()
            print('>>> training time needed in seconds', (t2-t1))

            print('>>> tuning results:')
            if result and 'base model performance' in result:
                base_model_perf = result['base model performance']
                print('\t * WER of base model:', base_model_perf['eval_wer'])
            if result and 'final model performance' in result:
                final_model_perf = result['final model performance']
                print('\t * WER of final model:', final_model_perf['eval_wer'])
            if result and 'best_model_dir' in result:
                best_model_dir = result['best_model_dir']
                print('\t * best model dir:', best_model_dir)

    print('>>> training completed, your run ID is:', runid)
    return runid
