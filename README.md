# NIST-Digits-using-Keras

Sample CNN training for NIST digits dataset using Keras. The training script has mini-batch generator in it.

## NIST Dataset

NIST special database 19 contains iages of handwritten characters. There are around 810,000 character images in the dataset. Nist dataset is available for download from the following url

* [NIST SD 19 home page](https://www.nist.gov/srd/nist-special-database-19)
* [Download the dataset segregated and used for training](https://drive.google.com/file/d/0B0LDJX3BuAYkSjA1VFk3M2tEYjA/view?usp=sharing)

The second link contains the data segregated into Digits, Capital letters and Lower case letters used for training. 

## Steps to train the dataset

# Step 1 - Prepare the dataset

1) Download the dataset from google drive link. 
2) Modify the paths in create_dataset.py and generate train and val set.

# Step 2 - Train

1) Modify the paths in train.py to point to appropriate train path
2) Modify the batch size to suit the gpu's memory. Batch size of 16 fits in 4gb of gpu memory.
3) Run the train.py and there will be a snapshot for evry epoch

