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

import datasets
import glob
import logging
import os
import pandas as pd
import pathlib
import random
import shutil

datasets.disable_caching()

import torch

# limit number of PyTorch CPU threads to avoid CPU thread contention happening during HF data processing (esp for audio feature extraction)
torch.set_num_threads(1)
print(">> Set number of PyTorch CPU threads: ", torch.get_num_threads())

TRAIN_METADATA_FILENAME = 'train_files.csv'
DEV_METADATA_FILENAME = 'dev_files.csv'
TEST_METADATA_FILENAME = 'test_files.csv'


def read_audio_dir(audio_recordings_dir, audio_type='wav'):
    """Read audio dir and expect audio files and prompt files."""
    prompt_files = glob.glob(os.path.join(audio_recordings_dir, '*.txt'))
    audio_files = glob.glob(os.path.join(audio_recordings_dir, '*.' + audio_type))
    if len(prompt_files) != len(audio_files):
        raise ValueError('Number of prompt files and audio files do not match. Ensure that correct audio_type is used (currently %s) and that each audio file has a prompt file.' % audio_type)
    assert len(prompt_files) == len(audio_files), 'number of prompt files and audio files do not match'

    recordings = [pathlib.Path(x).stem for x in audio_files]
    return recordings

def create_metadata_from_audio_dir(audio_recordings_dir, audio_type='wav', test_percentage=0.2, dev_percentage=0.2):
    """Create a dev and a train set from the audio recordings in the given folder.
    The audio_recordings_dir is expected to contain recording consisting of
    a audio file (ending in audio_type) and a text file (txt) for the prompt with the same name.
    
    Returns:
        dev_set: the dev set (HF)
        train_set: the train set (HF)
    """

    # creates splits and metadata from audio folder
    # dev can be 0, test cannot

    assert dev_percentage >= 0
    assert test_percentage >= 0
    assert dev_percentage + test_percentage < 0.5, "dev and test cannot exceed 50% of dataset"
    train_percentage = 1 - dev_percentage - test_percentage
    print(f">>>>>>>> train_percentage: {train_percentage}, dev_percentage: {dev_percentage}, test_percentage: {test_percentage}")

    logging.debug(f"Creating dataset from folder: {audio_recordings_dir}")
    uids = read_audio_dir(audio_recordings_dir, audio_type=audio_type)
    if not uids or len(uids) < 1:
        raise ValueError('No examples found, check audio_type and audio_recorder_dir (files are expected in data subdir)')

    test_entries = []
    dev_entries = []
    train_entries = []

    for uid in uids:
        audio_file = uid + '.' + audio_type
        prompt_file = os.path.join(audio_recordings_dir, uid + '.txt')
        with open(prompt_file, 'r') as f:
            prompt = f.read().strip()
        split = random.choices(['train', 'dev', 'test'], weights=[train_percentage, dev_percentage, test_percentage], k=1)[0]
        entry = pd.Series({'uid': uid, 'file_name': audio_file, 'text': prompt})
        if split == 'train':
            train_entries.append(entry)
        elif split == 'dev':
            dev_entries.append(entry)
        elif split == 'test':
            test_entries.append(entry)

    # write entries to files
    result = {}
    if train_entries:
        train_metadata_file = os.path.join(audio_recordings_dir, TRAIN_METADATA_FILENAME)
        pd.DataFrame(train_entries).to_csv(train_metadata_file, index=False)
        result['train_metadata_file'] = train_metadata_file
    if dev_entries:
        dev_metadata_file = os.path.join(audio_recordings_dir, DEV_METADATA_FILENAME)
        pd.DataFrame(dev_entries).to_csv(dev_metadata_file, index=False)
        result['dev_metadata_file'] = dev_metadata_file
    if test_entries:
        test_metadata_file = os.path.join(audio_recordings_dir, TEST_METADATA_FILENAME)
        pd.DataFrame(test_entries).to_csv(test_metadata_file, index=False)
        result['test_metadata_file'] = test_metadata_file

    # check validity
    if not train_entries:
        raise ValueError('No train entries found')
    return result


class ASRDataset:

    def delete_hf_dataset_cache(self):
        """HF datasets caches aggressively after map function. Delete this
        to prevent issues when re-processing the same dataset again."""
        cache_path = os.path.expanduser('~/.cache/huggingface/datasets')
        if os.path.exists(cache_path):
            print(f"Deleting existing hf dataset cache: ", cache_path)
            shutil.rmtree(cache_path)


    def __init__(self, audio_recordings_dir, audio_type='wav', 
                 init_splits_automatically=True, delete_hf_dataset_cache=True):
        """Create a dataset for ASR training.
        
        Args:
            audio_recordings_dir: path to directory containing audio files and prompts
            audio_type: audio file type (wav or mp3)
            init_splits_automatically: if True, initialized the dataset splits based automatically: if metadata files exist, use them; otherwise randomply split with default proportions.
        """
        if delete_hf_dataset_cache:
            self.delete_hf_dataset_cache()
        self.audio_recordings_dir = audio_recordings_dir
        self.audio_type = audio_type

        audio_files_count = len([f for f in os.listdir(audio_recordings_dir) if f.lower().endswith('.' + audio_type)])
        # for filename in os.listdir(audio_recordings_dir):
        #     if filename.lower().endswith('.' + audio_type):
        #         audio_files_count += 1
        print(f"Number of audio files in directory:", audio_files_count)

        # initialize automatically
        if init_splits_automatically:
            expected_metadata_files = [os.path.join(audio_recordings_dir, f) for f in [TRAIN_METADATA_FILENAME, DEV_METADATA_FILENAME, TEST_METADATA_FILENAME]]
            if not all([os.path.exists(f) for f in expected_metadata_files]):
                print("Metadata files not found. Creating from audio files.")
                self.init_from_simple_audio_file_dir()
            else:
                print("All expected metadata files found. Using them to create datasets.")
                self.init_from_audio_folder_with_metadata_files()
        else:
            print("WARN: Not fully initialized -- need to create splits by calling init_from_simple_audio_file_dir or init_from_audio_folder_with_metadata_files.")

    def init_from_simple_audio_file_dir(self, test_percentage=0.1, dev_percentage=0.1):
        """Creates a dataset including split into train/test/dev from a simple audio file folder.
        
        Folder is expected to include audio files (ending in audio_type) and a text file (txt) for the prompt with the same name.
        Splits are created randomly according to percentage provided.
        """

        result = create_metadata_from_audio_dir(self.audio_recordings_dir, self.audio_type, 
                                                test_percentage, dev_percentage)
        self.train_metadata_file = result['train_metadata_file']
        self.dev_metadata_file = result['dev_metadata_file']
        self.test_metadata_file = result['test_metadata_file']

        print(f"Created metadata files: {self.train_metadata_file}, {self.dev_metadata_file}, {self.test_metadata_file}")


    def init_from_audio_folder_with_metadata_files(self):
        """Creates a dataset fro a specificed audio file folder which contains metadata files for train/dev/test.
        
        Folder is expected to include audio files (ending in audio_type) and a text file (txt) for the prompt with the same name.
        Metadata files are expected to exist and be named train_files.csv, dev_files.csv, test_files.csv.
        """

        self.train_metadata_file = os.path.join(self.audio_recordings_dir, TRAIN_METADATA_FILENAME)
        if not os.path.exists(self.train_metadata_file):
            raise ValueError(f"Metadata file not found: {self.train_metadata_file}")

        self.test_metadata_file = os.path.join(self.audio_recordings_dir, TEST_METADATA_FILENAME)
        if not os.path.exists(self.train_metadata_file):
            raise ValueError(f"Metadata file not found: {self.train_metadata_file}")

        self.dev_metadata_file = os.path.join(self.audio_recordings_dir, DEV_METADATA_FILENAME)
        if not os.path.exists(self.train_metadata_file):
            raise ValueError(f"Metadata file not found: {self.train_metadata_file}")

        print(f"Reusing existing metadata files: {self.train_metadata_file}, {self.dev_metadata_file}, {self.test_metadata_file}")

    def make_hf_datasets(self, overwrite=False):
        if overwrite or not hasattr(self, 'hf_train_set'):
            self.make_hf_train_set()
            print("Created HF train set: ", self.hf_train_set)

        if overwrite or not hasattr(self, 'hf_dev_set'):
            self.make_hf_dev_set()
            print("Created HF dev set: ", self.hf_dev_set)

        if overwrite or not hasattr(self, 'hf_test_set'):   
            self.make_hf_test_set()
            print("Created HF test set: ", self.hf_test_set)

    def make_hf_train_set(self):
        print(">> making train set")
        self.hf_train_set = datasets.load_dataset(
            'audiofolder', data_dir=self.audio_recordings_dir, 
            metadata_filenames=[os.path.basename(self.train_metadata_file)], split='train').shuffle(seed=42).flatten_indices()
        print('done with train set, size:', len(self.hf_train_set))

    def make_hf_test_set(self):
        print(">> making test set from: ", self.audio_recordings_dir, os.path.basename(self.test_metadata_file))
        self.hf_test_set = datasets.load_dataset(
            'audiofolder', data_dir=self.audio_recordings_dir, 
            metadata_filenames=[os.path.basename(self.test_metadata_file)], split='test').shuffle(seed=42).flatten_indices()
        print('done with test set, size:', len(self.hf_test_set))
    
    def make_hf_dev_set(self):
        print(">> making dev set", self.audio_recordings_dir, '>>', os.path.basename(self.dev_metadata_file))
        self.hf_dev_set = datasets.load_dataset(
            'audiofolder', data_dir=self.audio_recordings_dir, 
            metadata_filenames=[os.path.basename(self.dev_metadata_file)], split='validation').shuffle(seed=42).flatten_indices()
        print('done with dev set, size:', len(self.hf_dev_set))

    def get_num_train_examples(self):
        return len(self.hf_train_set)

    def get_num_test_examples(self):
        return len(self.hf_test_set)

    def get_num_dev_examples(self):
        return len(self.hf_dev_set)        
    
    def show_dataset_stats(self):
        print("Dataset stats:")
        print(f"# train utterance: {self.get_num_train_examples()}")
        print(f"# test utterance: {self.get_num_test_examples()}")
        print(f"# dev utterance: {self.get_num_dev_examples()}")                