# Longformer-based hierarchical Labelwise attention Transformer

## Prerequisites
Restore [MIMIC-III v1.4 data](https://physionet.org/content/mimiciii/1.4/) into a Postgres database. 

## Download ClinicalplusXLNet
[ClinicalplusXLNet](https://unsw-my.sharepoint.com/:f:/g/personal/z5250377_ad_unsw_edu_au/Enw5NPgF2kFGrgqeE0LJLgABUKflITL9POL64S4uM7wJfg?e=IbyaNa)

## Training data preparation
python3 hilat/data/mimic3_data_preparer.py \
    --data_output_dir=your_data_dir \
    --pre_process_level=level_1 \
    --segment_text=True 
    
## Training model
1. Use run_coding.sh to train the model on TPU environment
2. Train on GPU

python3 hilat/run_coding.py config.json

