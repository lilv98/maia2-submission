# maia2-submission

# Requirements
* python == 3.11.5
* torch == 2.1.2
* numpy == 1.26.3
* pandas == 2.1.4
* tqdm == 4.65.0
* python-chess == 1.10.0


# Training

## Data Preparation
Modify the data path in 

    ./code/fetch_data.sh

and run:

    sh ./code/fetch_data.sh

## Start Training
We train our model with 2 Nvidia A100 GPUs and 48 AMD EPYC 7V13 CPUs.
The data preprocessing is integrated in the training process. To train Maia-2, simply run:

    cd ./code
    nohup python main.py --verbose 0 --data_root your_data_root >your_log_file.log 2>&1 &


# Inference

Download the model from [here](https://drive.google.com/file/d/1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc/view?usp=sharing).
To reproduce the main results, simply run:

    cd ./code
    python inference.py --model_path your_model_path