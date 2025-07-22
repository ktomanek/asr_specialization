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


# Convenience utils for model fine-tuning and evaluation
# based on https://huggingface.co/blog/fine-tune-whisper.

from dataclasses import dataclass
import evaluate
import inspect
import logging
import numpy as np
import os
from typing import Any, Dict, List, Union

from torch import Tensor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.models.whisper import tokenization_whisper

from asr_trainer import ASRTrainer
from asr_dataset import ASRDataset

TASK = 'transcribe'

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class WhisperTrainer(ASRTrainer):

    def __init__(self, language, asr_dataset: ASRDataset):
        
        # dataset
        print("Reading datasets")
        self.asr_dataset = asr_dataset
        self.asr_dataset.make_hf_datasets()
        
        if not hasattr(self.asr_dataset, 'hf_train_set') or not  hasattr(self.asr_dataset, 'hf_dev_set'):
            raise ValueError("Dataset must have train and dev split")
        
        # ensure language code is supported
        self.language = language
        print(f"Using language: {self.language}")
        languages_list = list(tokenization_whisper.TO_LANGUAGE_CODE.values())
        if not language in languages_list:
            raise ValueError(f"The specified language '{language}' is not supported by the Whisper models. Please choose from the supported languages.")

    def init_trainer(self, base_model_path_or_name, output_dir, **kwargs):
        """Loads processor, base model, prepare features and 
        creates hugging face trainer with specified features."""

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # create processor
        self.processor = WhisperProcessor.from_pretrained(
            base_model_path_or_name, language=self.language, task=TASK)
        print("Created processor...")

        # prepare features
        print("Extracting features for trainig data...")
        self.asr_dataset.hf_train_set = self._prepare_features_for_dataset(
            self.asr_dataset.hf_train_set, remove_audio_feature=True)
        print(">> training set finalized: ", self.asr_dataset.hf_train_set)

        print("Extracting features for dev data...")
        self.asr_dataset.hf_dev_set = self._prepare_features_for_dataset(
            self.asr_dataset.hf_dev_set, remove_audio_feature=True)
        print(">> dev set finalized: ", self.asr_dataset.hf_dev_set)

        # setup base model
        print("Setting up base model...")
        self.base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path_or_name)
        
        # setup trainer
        print("Setting up trainer...")
        sig_setup_base_model_for_training = inspect.signature(self._setup_base_model_for_training)
        params_setup_base_model_for_training = {
            key: value for key, value in kwargs.items() 
            if key in sig_setup_base_model_for_training.parameters
        }        
        self._setup_base_model_for_training(**params_setup_base_model_for_training)
        
        sig_setup_trainer = inspect.signature(self._setup_trainer)
        params_setup_trainer = {
            key: value for key, value in kwargs.items() 
            if key in sig_setup_trainer.parameters
        }        
        self._setup_trainer(**params_setup_trainer)

        print(">>> ready to training...")

    def train(self):
        self.huggingface_trainer.train()

    def evaluate(self):

        res = self.huggingface_trainer.evaluate(self.asr_dataset.hf_dev_set)
        return res
    
    def save_model(self, output_dir):
        self.huggingface_trainer.save_model(output_dir)
        
    def _prepare_features_for_example(self, example):
        example["input_features"] = self.processor.feature_extractor(example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"]).input_features[0]
        example["labels"] = self.processor.tokenizer(example["text"]).input_ids
        return example

    def _prepare_features_for_dataset(self, hf_dataset, remove_audio_feature=True):
        
        # limit number of cpu processors for feature extraction
        # (slowdown if we have too many relative to dataset size)
        l = int(len(hf_dataset) / 8)
        num_proc = min(32, l, os.cpu_count())
        

        print(f"Preparing features for dataset with {num_proc} processors")
        remove_cols = ['audio'] if remove_audio_feature else []
        hf_dataset = hf_dataset.map(
            lambda example: self._prepare_features_for_example(example), 
            remove_columns=remove_cols, writer_batch_size=1, num_proc=num_proc,
            # don't load from cache (may not always work)
            load_from_cache_file=False
            )
        return hf_dataset

    def _setup_base_model_for_training(self, update_encoder=True, update_decoder=True, update_proj=True):
        
        # ensure task and language for training
        self.base_model.generation_config.language = self.language
        self.base_model.generation_config.task = TASK
        self.base_model.generation_config.forced_decoder_ids = None
        self.base_model.config.forced_decoder_ids = None
        self.base_model.config.use_cache = False
        logging.info(f"Using Language: {self.language}")

        # TODO(ktomanek) add specaugment configuration
        # (currently only large models are pretrained with specaugment, but it can be 
        # added to smaller models as well, but will likely only make sense when enough 
        # training data is available).

        # which layers to tune
        self.base_model.model.encoder.requires_grad_(update_encoder)
        self.base_model.model.decoder.requires_grad_(update_decoder)
        self.base_model.proj_out.requires_grad_(update_proj)
        print(f"encoder params to update/total: {count_trainable_parameters(self.base_model.model.encoder)} / {self.base_model.model.encoder.num_parameters()}")
        print(f"decoder params to update/total: {count_trainable_parameters(self.base_model.model.decoder)} / {self.base_model.model.decoder.num_parameters()}")
        print(f"overall # trainable parameters: {count_trainable_parameters(self.base_model)}")
        print(f"overall # model parameters: {self.base_model.model.num_parameters()}")

    def _setup_trainer(
            self, 
            # checkpoint saving:
            save_every=50, max_saved_checkpoints=2,
            #
            # device args (if not specified, HF trainer will auto-detect and decide)
            use_mps_device=None, no_cuda=None,
            # more training stuff
            #use_fp16=False,  # set to False for CPU training
            report_to_tensorboard=True,      
            max_train_epochs=4,
            # training settings (good defaults for smaller datasets and models, overwrite if needed)
            train_batch_size=16,
            eval_batch_size=8,
            eval_on_start=False, eval_every=50,
            eval_max_gen_len=128,
            # for learning rate and scheduler
            learning_rate=1e-5, lr_scheduler='polynomial', lr_scheduler_warmup_steps=50, lr_scheduler_end_lr=1e-6, lr_scheduler_decay_power=4,
            ):
        """Sets up huggingface trainer with relevant parameters (default settings for small models and smaller datasets).

        For device settings:
        * If use_mps_device and no_cuda are unspecified, will let HF trainer decide.
        * If you have Apple Metal/MPS but want to use CPU, set: use_mps_device=False, no_cuda=True.
        * If specified device config is invalid (eg no_cuda=False if no GPU is present), HF trainer will fall back to auto-detect.
        
        """

        logdir = os.path.join(self.output_dir, 'logs')
        print(f"Model training logdir: {logdir}")

        reports = []
        if report_to_tensorboard:
            print(">>> Logging to tensorboard")
            reports = ["tensorboard"]

        # set device and FP16 use
        device_args = {}
        if use_mps_device is not None:
            device_args['use_mps_device'] = use_mps_device
            device_args['fp16'] = False
            print("Explicitly set use_mps_device:", use_mps_device)
        if no_cuda is not None:
            device_args['no_cuda'] = no_cuda
            device_args['fp16'] = not no_cuda
            print("Explicitly set no_cuda:", no_cuda)
        print(f"Use fp16: {device_args['fp16']}")

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            logging_dir=logdir,
            logging_steps=5,
            report_to=reports,
            include_num_input_tokens_seen=True,
            # device settings
            **device_args,
            push_to_hub=False,
            remove_unused_columns=False,
            #
            num_train_epochs=max_train_epochs,
            per_device_train_batch_size=train_batch_size,
            eval_on_start=eval_on_start,
            predict_with_generate=True,
            per_device_eval_batch_size=eval_batch_size,
            #
            eval_steps=eval_every,
            eval_strategy="steps",
            generation_max_length=eval_max_gen_len,
            #
            metric_for_best_model="wer",
            greater_is_better=False,
            
            lr_scheduler_type=lr_scheduler,
            # when lr_scheduler_type is 'constant', these are ignored as the constant scheduler doesn't use args
            # only applies to polynomial schedule
            lr_scheduler_kwargs={
                "lr_end": lr_scheduler_end_lr, # The final LR.  Crucial for polynomial decay.
                "power": lr_scheduler_decay_power, # for decay
                # we don't need to set the other arguments as they are already set in the args outside
                #"num_warmup_steps": WARMUP_STEPS, # The number of steps for the warmup phase.
                #"num_training_steps": MAX_STEPS, # The total number of training steps.
                #"lr_init": 1e-5 # we take the LR setting
            },

            learning_rate=learning_rate,
            warmup_steps=lr_scheduler_warmup_steps,
            #
            save_steps=save_every,
            save_strategy="steps",
            save_total_limit=max_saved_checkpoints,
            load_best_model_at_end=True,
        )
        print(f"Trainer using device: {training_args.device}")

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.base_model.config.decoder_start_token_id,
        )

        self.huggingface_trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.base_model,
            train_dataset=self.asr_dataset.hf_train_set,
            eval_dataset=self.asr_dataset.hf_dev_set,
            data_collator=data_collator,
            compute_metrics=lambda preds: self._compute_metrics(preds), 
            processing_class=self.processor
        )

    def _compute_metrics(self, pred):
        """Adds WER and CER (both example-averagered).

        We cap the per-example WER and CER at 1.0 and then average across all examples.
        
        Motivation to calculate a per-example average (as opposed to overall WER) as this 
        is better suited in extremely high WER scenarios (eg, when adapting to a new language)
        where model is likely to extremely hallucinate and return repetitions up to the
        token max count. Standard, corpus WER does capture improvements in this scenario well.
        """

        transcript_normalizer = BasicTextNormalizer()
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_strs = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wers = []
        cers = []
        for pred_str, label_str in zip(pred_strs, label_strs):
            p = transcript_normalizer(pred_str)
            l = transcript_normalizer(label_str)
            wer = wer_metric.compute(predictions=[p], references=[l])
            cer = cer_metric.compute(predictions=[p], references=[l])
            wers.append(wer)
            cers.append(cer)

        wer = np.mean([min(1.0,x) for x in wers])
        cer = np.mean([min(1.0,x) for x in cers])
        return {"wer": wer, "cer": cer}

