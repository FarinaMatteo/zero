#!/bin/bash

# parameters for the script
SET_ID="A" # dataset identifier, have a look at data/datautils.py
TEMPLATES_FLAG="" # "--templates" or empty string "". If "--templates", uses an ensemble of textual templates for CLIP (+Ensemble in the article)
MAPLE_FLAG="" # "--maple" or empty string "". If "--maple", uses a MaPLe initialization for CLIP. This and "--templates" should be mutually exclusive.
RESULTS_DIR=results # folder where results are saved
CTX="a_photo_of_a" # underscore separated hard prompt for CLIP (I've always used this one!)
SEED=1 # random seed (can only be 1,2 or 3 for MaPLe, any number will work otherwise)

if [[ ${TEMPLATES_FLAG}!="--templates" && ! -z "${TEMPLATES_FLAG}" ]] ; then
    echo "Please set the variable TEMPLATES_FLAG to either '--templates' or ''."
    exit
fi

if [[ ${MAPLE_FLAG}!="--maple" && ! -z "${MAPLE_FLAG}" ]] ; then
    echo "Please set the variable MAPLE_FLAG to either '--maple' or ''."
    exit
fi

# launch :)
python run.py \
--tuner Zero \
--arch ViT-B/16 \
--ctx_init ${CTX} \
--set_id ${SET_ID} \
${TEMPLATES_FLAG} \
${MAPLE_FLAG} \
--results_dir ${RESULTS_DIR} \
--seed ${SEED}