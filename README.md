# SkullCTSegmentation

 This repository contains a Pytorch implementation of semantic segmentation tasks for a multi-center skull CT dataset.

 This repository is still a work in progress. If there is any trouble, please let the person in charge know.  

<br>

## **Getting Started**

All commands and explanations in the sections below assume that the user is in a terminal and works on the directory as the repository folder containing each of its subfolders.
It also expects you to have downloaded or placed the target dataset in the working directory.
<!-- Consider your current working directory as the root directory.    -->

----

<br>  

## **Training**

To train the model, run the following command:

```bash
python train.py
```

The training script will automatically load the dataset and train the model. The training script will save the model weights in the `./weights` directory.
