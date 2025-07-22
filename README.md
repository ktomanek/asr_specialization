# Specialize ASR Models

This toolkit provides easy to use fine-tuning for ASR models (currently supported are whisper models, NVIDIA Nemo and Moonshine ASR currently WIP) with the goal to personalize or uber-specialize ASR models so as to be able to use smaller models with good performance.

The idea is that by specialization, smaller models will provide sufficient quality which is otherwise achieved by much larger models. Large models are inherently hard to run on-device.

The fine-tuning setup here focusses on (a) small(er) data sets and (b) small(er) models. Default tuning parameters are optimized for scenarios with 100-300 audio examples per.

This toolkit includes:

* wrappers for common ASR models for tuning
* convenience methods for creating a audio dataset
* tooling and examples to train in different environments:
   * locally
   * deployment of a training server with REST API to receive training calls
   * training on on-demand GPUs from providers like Modal


## Installation

* pip install -r requirements.txt
* make sure, ffmpeg is installed

## Datasets

`lib/asr_dataset.py` provides a convenience wrapper to turn a folder with audio files into a dataset that can be used for training and evaluation of ASR models. This includes the generation of splits if not available.

Minimally required structure of an **audio folder**:

* separate audio file per recording (limit to 30 seconds); naming convention: `<UID>.wav` (other audio formats, like eg `mp3` also supported)
* per audio file, one file with the respective transcript; naming convention: `<UID>.txt`

Optionally, you can add also 3 metadata files that describe the splits:

* `[train|dev|test]_files.csv`
* each file, needs to consist of UID, audio filename and transcript, like so:

```
uid,file_name,text
69,69.mp3,Dreaming of travel to distant lands.
54,54.mp3,Looking so pretty in that dress.
182,182.mp3,Reading a captivating novel.
[...]
```

* `lib/asr_dataset.py` will create these metadata files from the audio and transcript files, if requested or not existing:

```
my_dataset = ASRDataset(audio_type='mp3', audio_recordings_dir=..., init_splits_automatically=True)
```

* audio folder
* splits
* reuse



## Training Tooling

There are several libraries for fine-tuning different ASR models (eg, `whisper_asr_trainer.py`) in `lib`.

Note that the default configuration for the training parameters is oriented towards small-scale fine-tuning, largely with acoustic adaptation in mind (ie, to adapt to a specific language or non-standard speech on rather small datasets). For example, per default, only the encoder weights are updated, others are kept frozen (can be modified). Moreover, learning rates/scheduler and batch sizes are all preconfigured to be well suited for small datasets.


### Local Training

`src/examples/run_local_training.py` demonstrates how to run the training library.

### Training Server

`src/server/training_server.py` is an implementation of a Flask-based training server. This server is meant to be run on a machine with sufficient compute resources (typically GPUs, machines with Apple Silican/M2+ chips work well for small models). As such, it could also be set up on a serverless machine via cloud providers like RunPod, Google Cloud Run etc.

#### Overview

* trainer waits for training requests from user
* forks for training (blocks for other processes during that time)
* status info available via endpoint (shows where in training it is, link to tensorboard, logdir etc)
* tensorboard started during training on other port as a separate process
* once done, tuned model can be downloaded

#### Usage

* example command to send a training request to the server (will return the training_id):
    `curl -X POST -F "base_model_name=openai/whisper-tiny" -F "data=@/tmp/data.tar.gz" -F "epochs=3" http://127.0.0.1:5553/training`
    * audio folder needs to contain audio (mp3) and text files (txt), each with the same base filename (eg `123.mp3` and `123.txt`), where the text file contains the transcript and the audio file is expected to be a 16khz recording of the phrase. 
    * Create a `tar.gz` file of your audio  (only include the files, not the folder itself): 
        * `tar -czvf data.tar.gz -C my_audio_folder .` 
    * server will provide ID of training job, which can be used to later download the model
* to check status of training process:
    `curl http://127.0.0.1:5553/status`
* download the model when training has finished:
    `curl 'http://127.0.0.1:5553/download_model?training_id=1739926871' -o /tmp/model.tar.gz`


For example, when training is running, the output of the status request looks like this, including a link to the tensorboard instance started locally.

``` 
{
"progress": "running base model evaluation",
"status": "busy",
"tensorboard_url": "http://localhost:6006/",
"training_dir": "/tmp/training_app/training/1743450314",
"training_id": "1743450314"
}
```

#### Example: Hosting the training server on RunPod

* When running the server on for example RunPod, you can open the port (here 5553) in your pod template and then start the server there (need to set `SERVER_HOSTNAME='0.0.0.0'`) and then use the URL runpod assigned, eg like this for the request:
```curl -X POST -F "base_model_name=openai/whisper-tiny" -F "data=@/tmp/data.tar.gz" -F "epochs=5" -F "language=en" https://7fd05eukwj0oza-5553.proxy.runpod.net/training```

* And similarly to download the model:
```curl 'https://7fd05eukwj0oza-5553.proxy.runpod.net/download_model?training_id=1740011233' -o /tmp/1740011233_model.tar.gz```



### On-Demand GPU Training on Modal

* This package also provides `on-demand` remote training implementations, where training can be kicked off from a local device to be executed on a server/instance with better resources/accelerators. 
* Currently on-demand training implementations are available for [Modal](https://modal.com/) on, but adapting this to other providers (like Baseten) should be straightforward.


Modal is a serverless platform that enables on-demand GPU usage by allowing users to run code on GPUs only when needed, automatically spinning them down during periods of inactivity to save costs.

To be able to use this code, you need to set up Modal locally. See `Getting Started` under `https://modal.com/docs/guide`. Once this is done, you can test the on-demand training with `trainer_on_model.py`.
* Fine-tuning and personalizing small models via Modal is quite affordable. Assuming around 200-500 short training examples, 2-10 epochs should be sufficient for training. 
* Many of my experiments can be done within 15 minutes on L4 GPUs, hence amounting to ~ $ 0.2 per adapted model.
* Modal generously gives $30 of free credits/monthly which will allow you to do quite a lot of ASR model specialization!


#### Example Run

* `src/lib/run_ondemand_training.py` given and example for kicking of said on-demand training on modal, and once done downloading the final model as well as deleting the data from modal
* data is stored temporarily on Modal's storage in a volume and directory defined in `src/ondeman/trainer_on_modal.py`


Here's an example training run with configuration, time needed and cost:

* ~280 training examples (3-6 words each)
* whisper-small models
* training on Modal's L40S GPUs
* quality:
    * WER before tuning: 0.225
    * WER after tuning: 0.106
* time needed for training: 3.5 mins
* cost: < 0.25 USD



## Example: Effect of ASR Model Specialization

Below, I prototypically show impact of ASR specialization with this toolkit on a small dataset recorded by me:

### Setup

#### Dataset

* single speaker, English with heavy Germany accent
* short phrases, with words that will emphasize a German speaker's accent, some examples:

```
Around the corner is a coffee shop.
Is it raining outside?
Watching the birds fly so high.
Dreaming of travel to distant lands.
Thank you kindly for your generous help.
Under the willow tree we sat.
```

| split | utterances |
|-------|------|
| train | 166  |
| dev   | 74  |
| test  | 49  |

#### Procedure 

* in each case, we report WER (word error rate) on the dev set only here (test set skipped)
* training was done with full training set, adapting the encoder layers only (freeze decoder and proj layer)
* we train 10, measure performance on the test set every 0.5 epochs, and do early stopping is done based on dev set WER
* we report WER from before and from best adaptation along with training time
* training was done on a Mac M4 Pro (tiny and base model) and on Modal (small, medium, large model)

### Results

| model | # params | WER base model| WER specialized model |
|--|--|--|--|
| tiny    | 39M     | 28.6 | 17.6 |
| small   | 244M    | 19.1 | 8.4
| large-v3-turbo | 809 M | 8.3 | -

Above table shows the WER on the German accented dev set of the base model (ie, without any adaptation) and after specialization with the provided 166 training examples. By specializing the model, we basically reduce the WER in a comparable way to stepping up the model size significantly:

* After adaptation of the small model, it performs on par with the (unadapted) large-v3-turbo model, which has almost 3x the number of parameters.
* Similarly, after adaptation the tiny model performs even slightly better than the (unadapted) small model, having > 6x as many parameters.

These results are significant as they clearly show how a specialized model can outperform much larger models and hence be deployed on much less resource-powered hardware. For example, the whisper-tiny model runs easily on edge devices like a Raspberry Pi, while the small model requires a faster CPU for decent performance and the large-v3-turbo model effectively needs to be run on GPU with lots of memory.

### Disclaimer

This experiment does not aim to be a fully scientific experiment. One should repeat this experiment on a larger dataset with more different speakers and hence specializations. Moreover, the hyper-parameters used for training might not be the best possible ones and further improvements are quite possible.
