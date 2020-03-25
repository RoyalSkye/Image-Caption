#!/usr/bin/env python3

from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/Users/skye/docs/image_dataset/caption_dataset/dataset_coco.json',
                       image_folder='/Users/skye/docs/image_dataset',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/Users/skye/docs/image_dataset/dataset',
                       max_len=50)
