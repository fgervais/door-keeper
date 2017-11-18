#!/bin/bash

python build_image_data.py \
	--train_directory=./dataset/train/ \
	--output_directory=./dataset/ \
	--validation_directory=./dataset/valid/ \
	--labels_file=./dataset/labels.txt \
	--train_shards=1 \
	--validation_shards=1 \
	--num_threads=1

