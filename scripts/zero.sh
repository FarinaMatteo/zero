#!/bin/bash

# parameters for the script
SET_ID="A" # dataset identifier, have a look at data/datautils.py
RESULTS_DIR=results # folder where results are saved
CTX="a_photo_of_a" # underscore separated hard prompt for CLIP (I've always used this one!)
SEED=1 # random seed (can only be 1,2 or 3 for MaPLe, any number will work otherwise)

# if given, the following flags will be used
TEMPLATES_FLAG="" # "--templates" or empty string "". If "--templates", uses an ensemble of textual templates for CLIP (+Ensemble in the article)
MAPLE_FLAG="" # "--maple" or empty string "". If "--maple", uses a MaPLe initialization for CLIP. This and "--templates" should be mutually exclusive.


# launch :)
python run.py \
--tuner Zero \
--arch ViT-B/16 \
--pretrained openai \
--ctx_init ${CTX} \
--set_id ${SET_ID} \
${TEMPLATES_FLAG} \
${MAPLE_FLAG} \
--results_dir ${RESULTS_DIR} \
--seed ${SEED}