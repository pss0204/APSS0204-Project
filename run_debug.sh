#!/bin/bash

srun --exclusive --gres=gpu:1 \
	compute-sanitizer --tool memcheck ./main $@