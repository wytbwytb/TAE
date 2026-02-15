
# [EMNLP 2025] Token-Aware Editing of Internal Activations for Large Language Model Alignment

This repository provides the code for the paper [Token-Aware Editing of Internal Activations for Large Language Model Alignment](https://aclanthology.org/2025.emnlp-main.480.pdf). 
## Abstract

> Intervening the internal activations of large language models (LLMs) provides an effective inference-time alignment approach to mitigate undesirable behaviors, such as generating erroneous or harmful content, thereby ensuring safe and reliable applications of LLMs. However, previous methods neglect the misalignment discrepancy among varied tokens, resulting in deviant alignment direction and inflexible editing
strength. To address these issues, we propose a token-aware editing (TAE) approach to fully utilize token-level alignment information in the activation space, therefore realizing superior post-intervention performance. Specifically, a Mutual Information-guided Graph Aggregation (MIG) module first develops an MI-guided graph to exploit the tokens’ informative interaction for activation enrichment, thus improving alignment probing and facilitating intervention. Subsequently, Misalignment-aware Adaptive
Intervention (MAI) comprehensively perceives the token-level misalignment degree from token representation and prediction to guide the adaptive adjustment of editing strength, thereby enhancing final alignment performance. Extensive experiments on three alignment capabilities demonstrate the efficacy of TAE, notably surpassing baseline by 25.8% on the primary metric of truthfulness with minimal cost.

## Table of Contents
1. [Installation](#installation)
2. [TruthfulQA Evaluation](#truthfulqa-evaluation)
3. [Workflow](#workflow)
4. [How to Cite](#how-to-cite)


## Installation
In the root folder of this repo, run the following commands to set things up.
```
conda env create -f environment.yaml
conda activate tae
mkdir -p validation/results_dump/answer_dump
mkdir -p validation/results_dump/summary_dump
mkdir -p validation/splits
mkdir features
mkdir probes
```

## Workflow

(1) Get activations by running `bash get_activations.sh`. MI-enhanced head-wise activations, as well as activations for estimator training and probe validation, are stored in the `features` folder. Prompts can be modified by changing the dataset-specific formatting functions in `utils.py`. 

(2) Get estimators by running `python train_estimators.py --model_name llama3_instruct_8B --dataset_name tqa_mc2 --device 0`. The trained estimators will be stored in the `probes` folder.

(3) Get into the `validation` folder, then run `bash validate_single.sh` to perform single inference-time editing on the corresponding LLM. To try out multiple hyperparameter settings, you can run `bash validate_all.sh`. Read the code to learn about additional options.

(4) Get True and Info metrics by getting into the 'tools' folder and running `bash validate_judge_info_single.sh`. Also, you can run `bash validate_judge_info.sh` for all hyperparameter settings.

### Results

The results for llama3-instruct-8B from our paper are available in the `generation results` folder. You can use these to familiarize yourself with the evaluation code workflow.

## Additional datasets
The harmlessness and fairness evaluation datasets will be released soon.

## How to Cite

```

@inproceedings{wang2025token,
  title={Token-Aware Editing of Internal Activations for Large Language Model Alignment},
  author={Wang, Tianbo and Ma, Yuqing and Liao, Kewei and Yang, Chengzhao and Zhang, Zhange and Wang, Jiakai and Liu, Xianglong},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={9482--9520},
  year={2025}
}

```
