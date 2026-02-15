import os
import sys
sys.path.insert(0, "TruthfulQA")

import torch
import torch.nn as nn
import torch.nn.functional as F
import llama
import mistral
import baichuan
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial
from tools.draw_heatmap import draw_heatmap

from truthfulqa import utilities, models, metrics
import openai
from truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert

ENGINE_MAP = {
    'llama_7B': '/path/to/your/workdir/huggingface/decapoda-research-llama-7B-hf', 
    'alpaca_7B': '/path/to/your/workdir/huggingface/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_instruct_8B': '/path/to/your/workdir/huggingface/llama3-instruct-8b',
    'mistral_7B': '/path/to/your/workdir/huggingface/mistral-7b',
    'baichuan_chat_7B': '/path/to/your/workdir/huggingface/baichuan-7b-chat',
    'chatglm_6B': '/path/to/your/workdir/huggingface/chatglm-6b',
}

from truthfulqa.utilities import (
    format_prompt,
    format_prompt_with_answer_strings,
    split_multi_answer,
    format_best,
    find_start,
)
from truthfulqa.presets import preset_map, COMPARE_PRIMER
from truthfulqa.models import find_subsequence, set_columns, MC_calcs
from truthfulqa.evaluate import format_frame, data_to_dict
import random


def load_nq(path):
    # dataset = load_dataset("OamPatel/iti_nq_open_val")["validation"]
    dataset = load_from_disk(path)
    df = pd.DataFrame(columns=["Question", "Best Answer", "Correct Answers", "Incorrect Answers"])
    for row in tqdm(dataset):
        new_row = pd.DataFrame({"Question": [str(row["question"])], 
                                "Best Answer": random.sample([str(_) for _ in row["answer"]],1),
            "Correct Answers": [';'.join([str(_) for _ in row["answer"]])], "Incorrect Answers": [str(row["false_answer"])]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def load_triviaqa():
    dataset = load_dataset("OamPatel/iti_trivia_qa_val")["validation"]
    df = pd.DataFrame(columns=["question", "answer", "false_answer"])
    for row in dataset:
        new_row = pd.DataFrame({"question": [row["question"]], "answer": [[_ for _ in row["answer"]['aliases']]], "false_answer": [row["false_answer"]]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tokenized_tqa(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

def tokenized_tqa_with_start(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_starts = []
    
    # llama
    starts_token_context = tokenizer('Q: A: ', return_tensors='pt').input_ids[:, 1:-1]
    starts_token = starts_token_context[:, 2:]  # 只取 'A: ' 的部分
    
    # baichuan
    # starts_token = tokenizer('Q: A:', return_tensors='pt').input_ids[:, 2:] 
    # decoded_starts_token_context = tokenizer.decode(starts_token[0])

    # # chatglm
    # starts_token_context = tokenizer('Q: A: ', return_tensors='pt').input_ids[:, 1:-1]
    # starts_token = starts_token_context[:, 3:]  # 只取 'A: ' 的部分
    
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            
            # idx = prompt.find('A: ') + 3
            # all_starts.append(idx)
            # starts_token = tokenizer('A: ', return_tensors = 'pt').input_ids
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids

            # decoded_prompt = tokenizer.decode(prompt[0])
            # decoded_start_prompt = tokenizer.decode(starts_token[0])
            # print(f"{decoded_prompt}\n{decoded_start_prompt}")
            # tokens = tokenizer.convert_ids_to_tokens(prompt[0])
            # token_ids = prompt[0].tolist()
            # for token, token_id in zip(tokens, token_ids):
            #     print(f"{token}: {token_id}")

            # idx = find_sublist_index(prompt[0], starts_token[0])
            # idx_e = find_sublist_index(prompt[0], starts_token[0]) + len(starts_token[0])
            
            all_prompts.append(prompt)
            all_labels.append(label)
            
            all_starts.append(find_sublist_index(prompt[0], starts_token[0]) + len(starts_token[0]))
    
    return all_prompts, all_labels, all_starts

def find_sublist_index(a, b):
    for i in range(len(a) - len(b) + 1):
        if torch.equal(a[i:i+len(b)], b):
            return i
    return None

def find_sublist_index_last(a, b):
    for i in range(len(a) - len(b), 0, -1):
        if torch.equal(a[i:i+len(b)], b):
            return i
    return None

def tokenized_tqa_gen_end_q(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

def tokenized_tqa_gen(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories


def get_llama_activations_bau(model, prompt, device): 

    # HEADS = [f"transformer.encoder.layers.{i}.self_attention.head_out" for i in range(model.config.num_hidden_layers)]
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def get_llama_logits(model, prompt, device): 

    model.eval()
    with torch.no_grad(): 
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits

def save_probes(probes, path): 
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def load_probes(path): 
    """loads a list of sklearn lr probes from path"""
    with open(path, 'rb') as f: 
        probes = pickle.load(f)
    return probes

# -- TruthfulQA helper functions -- # 

def tqa_run_answers_ours(frame, engine, tag, preset, model=None, tokenizer=None, verbose=True, device=None, cache_dir=None, interventions={}, intervention_fn=None, instruction_prompt=True, many_shot_prefix=None):

    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""

    if tag not in frame.columns:
        frame[tag] = ''

    frame[tag].fillna('', inplace=True)
    frame[tag] = frame[tag].astype(str)

    # get tokens for ending sequence
    seq_start = np.array(tokenizer('A:')['input_ids'])
    seq_end = np.array(tokenizer('Q:')['input_ids'])

    tokens = []
    for idx in frame.index: 
        if pd.isnull(frame.loc[idx, tag]) or not len(frame.loc[idx, tag]):
            prompt = format_prompt(frame.loc[idx], preset, format='general')
            prefix = ''
            if instruction_prompt:  # from Ouyang et al. (2022) Figure 17, followed by LLaMA evaluation, and then followed by us
                prefix += 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.' + '\n\n'
            if many_shot_prefix is not None:
                prefix += many_shot_prefix + '\n\n'
            prompt = prefix + prompt            
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
            tokens.append(input_ids)

    # --- intervention code --- #
    def id(head_output, layer_name): 
        return head_output

    if interventions == {}: 
        intervene = id
        layers_to_intervene = []
    else: 
        intervene = partial(intervention_fn, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #

    sequences = []
    with torch.no_grad():
        for idx, input_ids in enumerate(tqdm(tokens)):
            max_len = input_ids.shape[-1] + 50

            # --- intervention code --- #

            with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret: 
                input_ids = input_ids.to(device)
                model_gen_tokens = model.generate(input_ids, top_k=1, max_length=max_len, num_return_sequences=1, use_uncertainty=True)[:, input_ids.shape[-1]:]
            
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
            model_gen_str = model_gen_str.strip()

            try: 
                # remove everything after 'Q:'
                model_gen_str = model_gen_str.split("Q:")[0].strip()
                # keep everything after A: 
                model_gen_str = model_gen_str.split("A:")[1].strip()
            except: 
                pass

            if verbose: 
                print("MODEL_OUTPUT: ", model_gen_str)
            
            frame.loc[idx, tag] = model_gen_str
            sequences.append(model_gen_str)

            # --- intervention code --- #

    if device:
        torch.cuda.empty_cache()

    return frame

def tqa_run_probs(frame, engine, tag, preset, model=None, tokenizer=None, verbose=True, device=None, cache_dir=None, 
                  interventions={}, intervention_fn=None, instruction_prompt=True, many_shot_prefix=None, 
                  probes=None):

    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""

    set_columns(tag, frame)

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(engine, return_dict_in_generate=True, cache_dir=cache_dir).to(device)
        model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(engine, cache_dir=cache_dir)

    with torch.no_grad():
        for idx in tqdm(frame.index):
            if pd.isnull(frame.loc[idx, '{0} lprob max'.format(tag)]):

                # check that answer exists
                if pd.isnull(frame.loc[idx, INCORRECT_COL]):
                    warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                    continue
                if not len(frame.loc[idx, INCORRECT_COL]):
                    warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                    continue

                # reference answers
                ref_best = format_best(frame.loc[idx, BEST_COL])
                ref_true = split_multi_answer(frame.loc[idx, ANSWER_COL])
                ref_false = split_multi_answer(frame.loc[idx, INCORRECT_COL])

                scores_true = []
                scores_false = []

                input_prompt = format_prompt(frame.loc[idx], preset, format='general')
                if many_shot_prefix is not None:
                    input_prompt = many_shot_prefix + input_prompt
                if instruction_prompt:
                    input_prompt = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.' + '\n\n' + input_prompt
                
                # --- intervention code --- #
                def id(head_output, layer_name): 
                    return head_output

                if interventions == {}: 
                    layers_to_intervene = []
                else: 
                    layers_to_intervene = list(interventions.keys())
                # --- intervention code --- #

                for temp_ans in ref_true:
                    # append the current answer choice to the prompt
                    prompt = format_prompt_with_answer_strings(frame.loc[idx, 'Question'],
                                                               temp_ans,
                                                               preset,
                                                               format='general')
                    if many_shot_prefix is not None:
                        prompt = many_shot_prefix + prompt
                    if instruction_prompt:
                        prompt = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.' + '\n\n' + prompt
                    
                    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
                    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
                    start_edit_location = input_ids.shape[-1] + 3 # account for the "lnA: " which is 4 tokens. Don't have to worry about BOS token because already in prompt
                    
                    if interventions == {}: 
                        intervene = id
                    else: 
                        intervene = partial(intervention_fn, start_edit_location=start_edit_location)
                    if probes is None:
                        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret: 
                            outputs = model(prompt_ids, use_uncertainty=True)[0].squeeze(0)
                            # outputs = model.forward_with_uncertainty(prompt_ids)[0].squeeze(0)
                    else:  
                        outputs = model.forward_with_select(input_ids=prompt_ids, output_hidden_states=True, probes=probes)[0].squeeze(0)
                    
                    outputs = outputs.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                    outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                    prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                    # get logprobs for each token in the answer
                    log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                    log_probs = log_probs[3:]  # drop the '\nA:' prefix 

                    scores_true.append(log_probs.sum().item())

                for temp_ans in ref_false:
                    # append the current answer choice to the prompt
                    prompt = format_prompt_with_answer_strings(frame.loc[idx, 'Question'],
                                                               temp_ans,
                                                               preset,
                                                               format='general')
                    if many_shot_prefix is not None:
                        prompt = many_shot_prefix + prompt
                    if instruction_prompt: 
                        prompt = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.' + '\n\n' + prompt
                    
                    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
                    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    start_edit_location = input_ids.shape[-1] + 3 # account for the "lnA: " which is 4 tokens. Don't have to worry about BOS token because already in prompt
                    
                    
                    if interventions == {}:
                        intervene = id
                    else:
                        intervene = partial(intervention_fn, start_edit_location=start_edit_location)

                    if probes is None:
                        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret: 
                            outputs = model(prompt_ids, use_uncertainty=True)[0].squeeze(0)
                            # outputs = model.forward_with_uncertainty(prompt_ids)[0].squeeze(0)
                    else:  
                        outputs = model.forward_with_select(input_ids=prompt_ids, output_hidden_states=True, probes=probes)[0].squeeze(0)
                    
                    outputs = outputs.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                    outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                    prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                    # get logprobs for each token in the answer
                    log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                    log_probs = log_probs[3:] # drop the '\nA:' prefix

                    scores_false.append(log_probs.sum().item())

                MC_calcs(tag, frame, idx, scores_true, scores_false, ref_true, ref_best)

    if device:
        torch.cuda.empty_cache()

    return frame

def run_ce_loss(model_key, model=None, tokenizer=None, device='cuda', interventions={}, intervention_fn=None, num_samples=100): 

    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = load_dataset("/path/to/your/workdir/tae/openwebtext-10k")['train']
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    # define intervention
    def id(head_output, layer_name):
        return head_output
    
    if interventions == {}:
        layers_to_intervene = []
        intervention_fn = id
    else: 
        layers_to_intervene = list(interventions.keys())
        intervention_fn = partial(intervention_fn, start_edit_location=0)

    losses = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    with torch.no_grad(): 
        for i in tqdm(rand_idxs):

            input_ids = owt[i]['input_ids'][:, :128].to(device)
            
            with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
                loss = model(input_ids, labels=input_ids).loss
            
            losses.append(loss.item())
    
    return np.mean(losses)

def run_kl_wrt_orig(model_key, model=None, tokenizer=None, device='cuda', interventions={}, intervention_fn=None, num_samples=100, separate_kl_device=None): 

    assert 'llama' in model_key or 'alpaca' in model_key or 'vicuna' in model_key or 'mistral' in model_key, 'model must be llama model'

    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = load_dataset("/path/to/your/workdir/tae/openwebtext-10k")['train']
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    # define intervention
    def id(head_output, layer_name):
        return head_output
    
    if interventions == {}:
        layers_to_intervene = []
        intervention_fn = id
    else: 
        layers_to_intervene = list(interventions.keys())
        intervention_fn = partial(intervention_fn, start_edit_location=0)

    kl_divs = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()

    if separate_kl_device is not None: 
        orig_model = llama.LLaMAForCausalLM.from_pretrained(ENGINE_MAP[model_key], torch_dtype=torch.float16, low_cpu_mem_usage=True)
        orig_model.to('cuda')

    with torch.no_grad(): 
        for i in tqdm(rand_idxs):
            input_ids = owt[i]['input_ids'][:, :128].to(device)

            if separate_kl_device is not None: 
                orig_logits = orig_model(input_ids.to('cuda')).logits.cpu().type(torch.float32)
            else: 
                orig_logits = model(input_ids).logits.cpu().type(torch.float32)
                
            orig_probs = F.softmax(orig_logits, dim=-1)

            with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
                logits = model(input_ids).logits.cpu().type(torch.float32)
                probs  = F.softmax(logits, dim=-1)
            
            kl_div = (orig_probs * (orig_probs / probs).log()).sum() / (input_ids.shape[-1] * input_ids.shape[-2])
            kl_divs.append(kl_div.item())

    return np.mean(kl_divs)

def alt_tqa_evaluate(models, metric_names, input_path, output_path, summary_path, device='cpu', verbose=False, preset='qa', 
                     interventions={}, intervention_fn=None, cache_dir=None, separate_kl_device=None, instruction_prompt=True, many_shot_prefix=None, 
                     judge_name=None, info_name=None, probes=None): 
    """
    Inputs:
    models: a dictionary of the form {model_name: model} where model is a HF transformer # TODO: doesn't work with models other than llama right now
    metric_names: a list of metric names to evaluate (ex: ['mc', 'judge', 'info', 'bleu'])
    input_path: where to draw TruthfulQA questions from
    output_path: where to store model outputs and full metric outputs
    summary_path: where to store metric summaries
    interventions: a dictionary of the form {layer_name: [(head, direction, projected_mean, projected_std)]}
    intervention_fn: a function that takes in a head output and a layer name and returns the intervened output

    Outputs a pd dataframe with summary values
    """

    questions = utilities.load_questions(filename=input_path)

    print("ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET")
    import os
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    for mdl in models.keys(): 

        # gpt-3
        if mdl in ['ada', 'babbage', 'curie', 'davinci']:  # gpt-3 models
            try:
                models.run_GPT3(questions, mdl, mdl, preset)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs_GPT3(questions, mdl, mdl, preset=preset)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

        # gpt-2
        if mdl in ['gpt2', 'gpt2-xl']:
            try:
                print(questions)
                questions = models.run_answers(questions, mdl, mdl, preset, device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs(questions, mdl, mdl, preset=preset, device=device, cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

        # llama
        if mdl in ['llama_7B', 'alpaca_7B', 'vicuna_7B', 'llama2_chat_7B', 'llama2_chat_13B', 'llama2_chat_70B', 'llama3_instruct_8B', 'mistral_7B', 'baichuan_chat_7B']: 

            assert models[mdl] is not None, 'must provide llama model'
            llama_model = models[mdl]
            if mdl in ['llama3_instruct_8B']:
                llama_tokenizer = llama.LlamaTokenizerFast.from_pretrained(ENGINE_MAP[mdl])
            elif mdl in ['mistral_7B']:
                llama_tokenizer = llama.LlamaTokenizer.from_pretrained(ENGINE_MAP[mdl], use_fast=False)
            elif mdl in ['baichuan_chat_7B']:
                llama_tokenizer = baichuan.BaichuanTokenizer.from_pretrained(ENGINE_MAP[mdl])
            else:
                llama_tokenizer = llama.LlamaTokenizer.from_pretrained(ENGINE_MAP[mdl])
            
            if 'judge' in metric_names or 'info' in metric_names:
                questions = tqa_run_answers_ours(questions, ENGINE_MAP[mdl], mdl, preset, model=llama_model, tokenizer=llama_tokenizer,
                                device=device, cache_dir=cache_dir, verbose=verbose,
                                interventions=interventions, intervention_fn=intervention_fn, instruction_prompt=instruction_prompt, many_shot_prefix=many_shot_prefix)

            utilities.save_questions(questions, output_path)

            if 'mc' in metric_names:
                questions = tqa_run_probs(questions, ENGINE_MAP[mdl], mdl, model=llama_model, tokenizer=llama_tokenizer, preset=preset, device=device, cache_dir=cache_dir, verbose=False, 
                                          interventions=interventions, intervention_fn=intervention_fn, instruction_prompt=instruction_prompt, many_shot_prefix=many_shot_prefix, 
                                          probes=probes)
                utilities.save_questions(questions, output_path)
        
        # gpt-neo
        if mdl in ['neo-small', 'neo-med', 'neo-large']:
            try:
                models.run_answers(questions, ENGINE_MAP[mdl], mdl, preset,
                                   device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs(questions, ENGINE_MAP[mdl], mdl, preset=preset, device=device,
                                     cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print("ERROR")
                print(err)

        # unifiedqa
        if mdl in ['uqa-small', 'uqa-base', 'uqa-large', 'uqa-3b']:
            try:
                models.run_UnifQA(questions, ENGINE_MAP[mdl], mdl, preset, device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs_T5(questions, ENGINE_MAP[mdl], mdl, preset, device=device, cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

    # save all
    utilities.save_questions(questions, output_path)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2', 'MC3',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')

    # calculate cross entropy loss on owt and kl wrt to original unedited on owt
    results['CE Loss'] = np.nan
    results['KL wrt Orig'] = np.nan

    # for model_key in models.keys(): 
    #     # if model_key not in questions.columns:
    #     #     warnings.warn("Answers missing for {0}!".format(model_key), stacklevel=2)
    #     #     continue
    #     if 'llama' in model_key or 'alpaca' in model_key or 'vicuna' in model_key or 'mistral' in model_key: 
    #         ce_loss = run_ce_loss(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn)
    #         kl_wrt_orig = run_kl_wrt_orig(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn, separate_kl_device=separate_kl_device)

    #     results.loc[model_key, 'CE Loss'] = ce_loss
    #     results.loc[model_key, 'KL wrt Orig'] = kl_wrt_orig

    # save results
    results.to_csv(summary_path, index=False)
    
    return results



def tqa_evaluate(models, metric_names, input_path, output_path, summary_path, device='cpu', verbose=False, preset='qa', interventions={}, intervention_fn=None, cache_dir=None, separate_kl_device=None, instruction_prompt=True, many_shot_prefix=None, judge_name=None, info_name=None): 
    """
    Inputs:
    models: a dictionary of the form {model_name: model} where model is a HF transformer # TODO: doesn't work with models other than llama right now
    metric_names: a list of metric names to evaluate (ex: ['mc', 'judge', 'info', 'bleu'])
    input_path: where to draw TruthfulQA questions from
    output_path: where to store model outputs and full metric outputs
    summary_path: where to store metric summaries
    interventions: a dictionary of the form {layer_name: [(head, direction, projected_mean, projected_std)]}
    intervention_fn: a function that takes in a head output and a layer name and returns the intervened output

    Outputs a pd dataframe with summary values
    """

    questions = utilities.load_questions(filename=input_path)

    print("ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET")
    import os
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    for mdl in models.keys(): 
        # gpt-3
        if mdl in ['ada', 'babbage', 'curie', 'davinci']:  # gpt-3 models
            try:
                models.run_GPT3(questions, mdl, mdl, preset)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs_GPT3(questions, mdl, mdl, preset=preset)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

        # gpt-2
        if mdl in ['gpt2', 'gpt2-xl']:
            try:
                print(questions)
                questions = models.run_answers(questions, mdl, mdl, preset, device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs(questions, mdl, mdl, preset=preset, device=device, cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

        # llama
        if mdl in ['llama_7B', 'alpaca_7B', 'vicuna_7B', 'llama2_chat_7B', 'llama2_chat_13B', 'llama2_chat_70B']: 

            assert models[mdl] is not None, 'must provide llama model'
            llama_model = models[mdl]
            llama_tokenizer = llama.LlamaTokenizer.from_pretrained(ENGINE_MAP[mdl])
            
            if 'judge' in metric_names or 'info' in metric_names:
                questions = tqa_run_answers_ours(questions, ENGINE_MAP[mdl], mdl, preset, model=llama_model, tokenizer=llama_tokenizer,
                                device=device, cache_dir=cache_dir, verbose=verbose,
                                interventions=interventions, intervention_fn=intervention_fn, instruction_prompt=instruction_prompt, many_shot_prefix=many_shot_prefix)

            utilities.save_questions(questions, output_path)

            if 'mc' in metric_names:
                questions = tqa_run_probs(questions, ENGINE_MAP[mdl], mdl, model=llama_model, tokenizer=llama_tokenizer, preset=preset, device=device, cache_dir=cache_dir, verbose=False, interventions=interventions, intervention_fn=intervention_fn, instruction_prompt=instruction_prompt, many_shot_prefix=many_shot_prefix)
                utilities.save_questions(questions, output_path)
        
        # gpt-neo
        if mdl in ['neo-small', 'neo-med', 'neo-large']:
            try:
                models.run_answers(questions, ENGINE_MAP[mdl], mdl, preset,
                                   device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs(questions, ENGINE_MAP[mdl], mdl, preset=preset, device=device,
                                     cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print("ERROR")
                print(err)

        # unifiedqa
        if mdl in ['uqa-small', 'uqa-base', 'uqa-large', 'uqa-3b']:
            try:
                models.run_UnifQA(questions, ENGINE_MAP[mdl], mdl, preset, device=device, cache_dir=cache_dir)
                utilities.save_questions(questions, output_path)
                if 'mc' in metric_names:
                    models.run_probs_T5(questions, ENGINE_MAP[mdl], mdl, preset, device=device, cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
            except Exception as err:
                print(err)

    for model_key in models.keys(): 

        for metric in metric_names: 
            if metric == 'mc':
                continue
            if metric == 'bleurt':
                try:
                    questions = metrics.run_BLEURT(model_key, questions, cache_dir=cache_dir)
                    utilities.save_questions(questions, output_path)
                except Exception as err:
                    print(err)
            elif metric in ['bleu', 'rouge']:
                try:
                    questions = metrics.run_bleu_and_rouge(model_key, questions)
                    utilities.save_questions(questions, output_path)
                except Exception as err:
                    print(err)
            elif metric in ['judge', 'info']:
                try:
                    if metric == 'judge':
                        questions = metrics.run_end2end_GPT3(model_key, 'GPT-judge', judge_name, questions, info=False)
                        utilities.save_questions(questions, output_path)
                    else:
                        questions = metrics.run_end2end_GPT3(model_key, 'GPT-info', info_name, questions, info=True)
                        utilities.save_questions(questions, output_path)
                except Exception as err:
                    print(err)
            else:
                warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)

    # save all
    utilities.save_questions(questions, output_path)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')

    # calculate cross entropy loss on owt and kl wrt to original unedited on owt
    results['CE Loss'] = np.nan
    results['KL wrt Orig'] = np.nan

    for model_key in models.keys(): 
        # if model_key not in questions.columns:
        #     warnings.warn("Answers missing for {0}!".format(model_key), stacklevel=2)
        #     continue
        if 'llama' in model_key or 'alpaca' in model_key or 'vicuna' in model_key:
            ce_loss = run_ce_loss(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn)
            kl_wrt_orig = run_kl_wrt_orig(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn, separate_kl_device=separate_kl_device)

        results.loc[model_key, 'CE Loss'] = ce_loss
        results.loc[model_key, 'KL wrt Orig'] = kl_wrt_orig

    # save results
    results.to_csv(summary_path, index=False)
    
    return results

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def train_probes(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, 
                 num_layers, num_heads):
    
    all_head_accs = []
    probes = []

    for activation in separated_head_wise_activations:
        if np.isnan(activation).any():
            print(1)

    # 准备probe训练数据（X,y），X为内部特征，y为真实与否标签0/1
    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
    
    # 对每一层layer、每个头部head均训练一个probe分类器
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
    
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)

            # 验证准确率
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np

from scipy.optimize import minimize
def get_top_heads(train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False):

    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []
    
    # sorted_all_head_accs = -np.sort(-all_head_accs_np, axis=1,)
    # draw_heatmap(sorted_all_head_accs * 100, np.array(range(0, num_heads)), np.array(range(0, num_layers)), 
    #              '/path/to/your/workdir/tae/figures/accs_allwords.jpg', size=6, xlabel='head index', ylabel='layers index')
    # topk_accs = -np.sort(-all_head_accs_np.reshape(num_heads*num_layers))[:num_to_intervene]
    # print(topk_accs, np.average(topk_accs))
    
    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
    
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes

def val_probes(seed, val_set_idxs, separated_head_wise_activations, separated_labels, probes, num_layers, num_heads, num_to_intervene, reverse=False):
    
    all_head_accs = []
    all_head_f1s = []
    all_head_pres = []
    all_head_recs = []

    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)

    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_val = all_X_val[:,layer,head,:]
            probe = probes[layer * num_layers + head]
    
            # clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            # y_pred = probe.predict(X_train)
            y_val_pred = probe.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            all_head_f1s.append(f1_score(y_val, y_val_pred))
            all_head_pres.append(precision_score(y_val, y_val_pred))
            all_head_recs.append(recall_score(y_val, y_val_pred))

    all_head_accs_np = np.array(all_head_accs)
    all_head_f1s_np = np.array(all_head_f1s)
    all_head_pres_np = np.array(all_head_pres)
    all_head_recs_np = np.array(all_head_recs)
    sorted_idx = np.argsort(all_head_accs_np)[::-1]
    for k in [8, 16, 24, 32, 40, 48]:
        topk_accs = all_head_accs_np[sorted_idx][:k]
        topk_f1s = all_head_f1s_np[sorted_idx][:k]
        topk_pres = all_head_pres_np[sorted_idx][:k]
        topk_recs = all_head_recs_np[sorted_idx][:k]
        print(k, np.average(topk_accs), np.average(topk_f1s), np.average(topk_pres), np.average(topk_recs))

    for k in [8, 16, 24, 32, 40, 48]:
        topk_accs = -np.sort(-all_head_accs_np.reshape(num_heads*num_layers))[:k]
        print(topk_accs, np.average(topk_accs))

    top_heads = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_heads]
    print(top_heads)
    return topk_accs, top_heads

def train_estimator(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads):
    all_head_accs = []
    estimators = []

    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)

    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
    
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            y_prob_pred = clf.predict_log_proba(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            estimators.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return estimators, all_head_accs_np

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions, args,
                           estimators=None): 

    interventions = {}
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = []
    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        else: 
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        t = np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interven = (head, direction.squeeze(), proj_val_std, args.beta)
        if estimators != None:
            loc = estimators[layer_head_to_flattened_idx(layer, head, num_heads)]
            interven = (head, direction.squeeze(), proj_val_std, args.beta, loc)
        interventions[f"model.layers.{layer}.self_attn.head_out"].append(interven)
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(interventions[f"model.layers.{layer}.self_attn.head_out"], key = lambda x: x[0])

    return interventions

def get_separated_activations(labels, head_wise_activations): 

    # separate activations by question
    dataset=load_dataset('/path/to/your/workdir/tae/truthful_qa_hf', 'multiple_choice')['validation']
    actual_labels = []
    for i in range(len(dataset)):
        actual_labels.append(dataset[i]['mc2_targets']['labels'])

    idxs_to_split_at = np.cumsum([len(x) for x in actual_labels])        

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    assert separated_labels == actual_labels

    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, idxs_to_split_at

def get_separated_activations_customized(labels, head_wise_activations, lens): 

    # separate activations by question
    # dataset=load_dataset('/path/to/your/workdir/tae/truthful_qa_hf', 'multiple_choice')['validation']
    # actual_labels = []
    # for i in range(len(dataset)):
    #     actual_labels.append(dataset[i]['mc2_targets']['labels'])

    idxs_to_split_at = np.cumsum(lens)[:-1]        

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    # assert separated_labels == actual_labels

    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, idxs_to_split_at


def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels): 

    com_directions = []

    for layer in range(num_layers): 
        for head in range(num_heads): 
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in usable_idxs], axis=0)
            usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def get_critical_words(dataset, tokenizer):
    critical_words = []
    answers = []
    for i in range(len(dataset)): 
        t = 0
        question = dataset[i]['question']
        q_tokens = tokenizer(question, return_tensors = 'pt').input_ids

        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']
        corrects = [choices[i] for i in range(len(choices)) if labels[i] == 1]
        corrects_idxs = [i for i in range(len(choices)) if labels[i] == 1]
        incorrects = [choices[i] for i in range(len(choices)) if labels[i] == 0]
        incorrects_idxs = [i for i in range(len(choices)) if labels[i] == 0]
        ### 去除begin 和 end token
        
        final_critials = get_critical_words_single(corrects, corrects_idxs, incorrects, incorrects_idxs, tokenizer)

        critical_words.extend(final_critials)
        answers.extend(choices)      
        
    return critical_words, answers
    

def get_critical_words_single(corrects, corrects_idxs, incorrects, incorrects_idxs, tokenizer):
    ## llama
    tokens_correct = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 3:] for c in corrects]
    set_tokens_correct = [set(tokens[0, :].tolist()) for tokens in tokens_correct]
    tokens_incorrect = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 3:] for c in incorrects]
    set_tokens_incorrect = [set(tokens[0, :].tolist()) for tokens in tokens_incorrect]
    
    ## baichuan
    # tokens_correct = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 2:] for c in corrects]
    # set_tokens_correct = [set(tokens[0, :].tolist()) for tokens in tokens_correct]
    # tokens_incorrect = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 2:] for c in incorrects]
    # set_tokens_incorrect = [set(tokens[0, :].tolist()) for tokens in tokens_incorrect]

    # ## chatglm
    # tokens_correct = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 4:] for c in corrects]
    # set_tokens_correct = [set(tokens[0, :].tolist()) for tokens in tokens_correct]
    # tokens_incorrect = [tokenizer('A: ' + c, return_tensors = 'pt').input_ids[:, 4:] for c in incorrects]
    # set_tokens_incorrect = [set(tokens[0, :].tolist()) for tokens in tokens_incorrect]

    #查看转换后的向量和对应的token
    # for i, input_ids in enumerate(tokens_correct):
    #     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    #     decoded_string = tokenizer.decode(input_ids[0])
    #     print(f"Original string: {'A' + corrects[i]}")
    #     print(f"Input IDs: {input_ids[0].tolist()}")
    #     print(f"Tokens: {tokens}")
    #     print(f"Decoded string: {decoded_string}")
    #     print("-" * 30)
    
    # for token in tokens_correct:
    #     if find_sublist_index(token[0], torch.tensor([853,5203], dtype=int)) != None:
    #         print(1)
    # for token in tokens_incorrect:
    #     if find_sublist_index(token[0], torch.tensor([853,5203], dtype=int)) != None:
    #         print(1)
    total = []
    for p, x in enumerate(set_tokens_correct):
        l = []
        for q, y in enumerate(set_tokens_incorrect):
            same = x.intersection(y)
            ### 此处得到的idx是在有begin 和 end token的情况下的idx,所以最小值为1
            same_idx_correct = [j + 1 for j, item in enumerate(tokens_correct[p][0].tolist()) if item in same]
            same_idx_incorrect = [j + 1 for j, item in enumerate(tokens_incorrect[q][0].tolist()) if item in same]
            diff_correct = set_tokens_correct[p] - same
            diff_incorrect = set_tokens_incorrect[q] - same
            diff_idx_correct = [j + 1 for j, item in enumerate(tokens_correct[p][0].tolist()) if item in diff_correct]
            diff_idx_incorrect = [j + 1 for j, item in enumerate(tokens_incorrect[q][0].tolist()) if item in diff_incorrect]
            l.append({'same_correct': same_idx_correct, 'same_incorrect': same_idx_incorrect,
                    'diff_correct': diff_idx_correct, 'diff_incorrect': diff_idx_incorrect})
        total.append(l)
    
    final_critials = [None] * (len(corrects_idxs) + len(incorrects_idxs))
    for x in range(len(total)):
        best_idx = 0
        max_same = len(total[x][0]['same_correct'])
        for y in range(len(total[0])):
            if len(total[x][y]['same_correct']) > max_same:
                max_same = len(total[x][y]['same_correct'])
                best_idx = y
        final_critials[corrects_idxs[x]] = {'same': total[x][best_idx]['same_correct'], 'diff': total[x][best_idx]['diff_correct']}
    
    for y in range(len(total[0])):
        best_idx = 0
        max_same = len(total[0][y]['same_incorrect'])
        for x in range(len(total)):
            if len(total[x][y]['same_incorrect']) > max_same:
                max_same = len(total[x][y]['same_incorrect'])
                best_idx = x
        final_critials[incorrects_idxs[y]] = {'same': total[best_idx][y]['same_incorrect'], 'diff': total[best_idx][y]['diff_incorrect']}
    
    return final_critials

