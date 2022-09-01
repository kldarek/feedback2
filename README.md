# 2nd place solution in the Efficiency Track
# Kaggle Feedback Prize - Predicting Effective Arguments

[Report Link](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx)
[Kaggle Inference Notebook Link](https://www.kaggle.com/code/thedrcat/hf-43bpseudo-infer-single-full-data-model/notebook?scriptVersionId=104069039)

This repository contains the code that was used to train the 2nd place solution in the Efficiency Track of Kaggle Feedback Prize - Predicting Effective Arguments competition. 

We followed a multi-stage approach to training the winning model, so the exact score may be hard to reproduce. I will highlight below the key elements of the solution:

1. MLM pre-training: notebooks/HF-pret-7.py
2. 1st stage models: notebooks/HF-43.ipynb
3. 2nd stage model distillation on pseudolabels: notebooks/HF-56-pseudolabels.ipynb
4. 2nd stage model finetuning: notebooks/HF-43pseudo2.ipynb
