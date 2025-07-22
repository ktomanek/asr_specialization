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


# Flask application running a server backend that can receive training commands.
# Only one model can be trained at a time -- app will return a "idle"/429 if a training job is already in progress.
#
# App has 3 endpoints:
# 1. `/training` (POST): Accepts training parameters (model name, data file, epochs, language) and starts a training job. It returns a training ID.
# 2. `/status` (GET): Returns the current status of the training job, including progress and tensorboard URL if available.
# 3. `/download_model` (GET): Allows downloading the trained model once training is complete, using the training ID.


from flask import Flask, request, jsonify, send_from_directory
import json
import os
import tarfile
from tensorboard_manager import TensorBoardManager
import threading
import time
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from asr_dataset import ASRDataset
from whisper_asr_trainer import WhisperTrainer


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("/tmp/modal_whisper_trainer.log", mode="a")
    ]
)

######################## configure server ######################
# use host='0.0.0.0' when running on remote server, eg runpod
SERVER_HOSTNAME='0.0.0.0'
SERVER_PORT=5553
TENSORBOARD_PORT = 6006

# Directory to store uploaded data
WORKING_DIR = '/tmp/training_app'
UPLOAD_FOLDER = os.path.join(WORKING_DIR, 'uploads')
TRAINING_FOLDER = os.path.join(WORKING_DIR, 'training')
################################################################

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAINING_FOLDER'] = TRAINING_FOLDER

# Global variables to track training status
training_lock = threading.Lock()
training_id = None
training_status = "idle"  # "idle" or training ID
training_progress = "idle"
training_thread = None
tensorboard_url = None


def get_model_dir(training_id):
    global TRAINING_FOLDER
    return os.path.join(TRAINING_FOLDER, str(training_id))

def get_results_dir(training_id):
    r = os.path.join(get_model_dir(training_id), 'results')
    os.makedirs(r, exist_ok=True)
    return r

def train_model(model_name, data_path, epochs, language, training_id_local):
    global training_status, training_progress, tensorboard_url
    try:
        logging.info(f"Starting training {training_id_local} for model {model_name} with {epochs} epochs...")
        training_status = "busy"
        training_progress = "preparing training"
        training_dir = get_model_dir(training_id_local)


        # read dataset
        my_dataset = ASRDataset(audio_type='mp3', audio_recordings_dir=data_path, init_splits_automatically=True)        

        # define trainer
        logging.info(f"setting trainign dir to: {training_dir}")
        os.makedirs(training_dir, exist_ok=True)
        whisper_tiny_trainer = WhisperTrainer(language=language, asr_dataset=my_dataset)
        whisper_tiny_trainer.init_trainer(base_model_path_or_name=model_name, 
                                        output_dir=training_dir,
                                        # device settings:
                                        use_mps_device=True, no_cuda=None,
                                        # which part of the model to update:
                                        update_encoder=True, update_decoder=False, update_proj=False,
                                        # trainer settings:
                                        report_to_tensorboard=True,
                                        save_every=50, max_saved_checkpoints=1,
                                        max_train_epochs=epochs,
                                        train_batch_size=8, 
                                        eval_batch_size=4,
                                        eval_on_start=False, eval_every=10)

        # start tensorboard
        training_progress = "starting tensorboard"
        tb_manager = TensorBoardManager(logdir=training_dir, port=TENSORBOARD_PORT)
        tb_manager.start()
        tensorboard_url = tb_manager.get_url()
        
        # evaluation of base model
        training_progress = "running base model evaluation"
        logging.info(f"evaluating base model...")
        res = whisper_tiny_trainer.evaluate()
        logging.info(f">>> base model eval result: {res}")
        with open(os.path.join(get_results_dir(training_id_local), 'basemodel_eval_result.txt'), 'w') as f:
            f.write(json.dumps(res))

        # start training
        training_progress = "running training"        
        whisper_tiny_trainer.train()

        # save best model
        training_progress = "saving best model"
        final_model_dir = get_results_dir(training_id_local)
        whisper_tiny_trainer.save_model(final_model_dir)        

        # run final eval
        training_progress = "running final model evaluation"
        logging.info(f"evaluating final...")
        res = whisper_tiny_trainer.evaluate()
        logging.info(f">>> final eval result: {res}")
        with open(os.path.join(get_results_dir(training_id_local), 'final_eval_result.txt'), 'w') as f:
            f.write(json.dumps(res))
        
        # stop tensorboard
        # tensorboard.program.shutdown_all()
        tb_manager.shutdown()
        logging.info("Tensorboard stopped. Training done.")
        ############## actual training ##############

        logging.info(f"Finished training {training_id_local}!, model is in {training_dir}")
        training_status = "idle" # Training is done.
        training_progress = "idle"
    except Exception as e:
        logging.info(f"Error during training {training_id_local}: {e}")
        training_status = "idle" # Even with an error, we are no longer training.
    finally:
        training_lock.release() # Release lock when done.
        training_thread = None # Reset the training thread.
        training_status = "idle"
        training_progress = "idle"


###### app #####
@app.route('/status', methods=['GET'])
def status():
    global training_id, training_progress, tensorboard_url, TENSORBOARD_PORT
    training_dir = get_model_dir(training_id)
    if training_status == 'busy':
        return jsonify({"status": training_status,
                        "progress": training_progress,
                        "training_id": training_id,
                        "tensorboard_url": tensorboard_url,
                        "training_dir": training_dir})
    else:
        return jsonify({"status": training_status})

@app.route('/training', methods=['POST'])
def training():
    global training_id, training_status, training_thread

    with training_lock: # Aquire the lock to check if already training
        if training_status != "idle":
            return jsonify({"message": "Service busy, come back later"}), 429  # HTTP 429 Too Many Requests

        model_name = request.form.get('base_model_name')
        if not model_name:
            return jsonify({"message": "Model name is required"}), 400

        uploaded_file = request.files.get('data')
        if not uploaded_file:
            return jsonify({"message": "Data file is required"}), 400

        try:
            epochs = int(request.form.get('epochs', 5)) # number of training epochs
            print("Number of epochs to train:", epochs)
        except ValueError:
            return jsonify({"message": "Invalid number of epochs"}), 400

        try:
            language = str(request.form.get('language', 'en'))
        except ValueError:
            return jsonify({"message": "Invalid language code"}), 400

        training_id = str(int(time.time())) # Simple ID based on timestamp
        training_status = training_id # Update status to current training ID

        # Save uploaded file
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], f"training_data_{training_id}.tar.gz")
        uploaded_file.save(data_path)

        # Extract the tar.gz file
        try:
            with tarfile.open(data_path, 'r:gz') as tar:
                members = tar.getmembers()
                tar.extractall(os.path.join(app.config['UPLOAD_FOLDER'], training_id), members=members) # Extract to a folder inside uploads named with the training id
            extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], training_id)

        except Exception as e:
            return jsonify({"message": f"Error extracting data: {e}"}), 500

        # start training thread
        training_thread = threading.Thread(target=train_model, args=(model_name, extracted_path, epochs, language, training_id))
        training_thread.start()

        return jsonify({"training_id": training_id, "message": "Training started, come back to check if it's ready"}), 200


@app.route('/download_model', methods=['GET'])
def get_model():
    """Downloads the trained model as a tar.gz file. Requires training_id as a query parameter."""
    training_id = request.args.get('training_id')  # Get training_id from query parameters
    if not training_id:
        return jsonify({"message": "Training ID is required"}), 400

    results_dir = get_results_dir(training_id)
    logging.info(f"getting model from: {results_dir}")
    if not os.path.exists(results_dir):
        return jsonify({"message": "Model not found for this training ID"}), 404

    try:
        # Create a tar.gz archive of the model directory
        model_archive_path = os.path.join(app.config['TRAINING_FOLDER'], f"{training_id}.tar.gz")
        with tarfile.open(model_archive_path, "w:gz") as tar:
            tar.add(results_dir, arcname=os.path.basename(results_dir)) # Important: arcname ensures the folder structure is preserved inside the archive
        logging.info("tar created, now sending...")

        # Serve the archive for download
        return send_from_directory(app.config['TRAINING_FOLDER'], f"{training_id}.tar.gz", as_attachment=True, mimetype="application/gzip")

    except Exception as e:
        logging.info(f"Error creating or sending model archive: {e}")
        return jsonify({"message": "Error creating or sending model archive"}), 500
    finally:
        # Clean up the archive after sending (optional, but good practice). You might want to keep the model directory itself.
        try:
            os.remove(model_archive_path)
        except Exception as e:
            print(f"Error removing archive {e}")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=SERVER_PORT, host=SERVER_HOSTNAME)


