import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import (get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, 
                   tokenized_tqa_gen_end_q, load_nq, tokenized_tqa_with_start, get_critical_words,
                   get_attention_modules, load_probes, rearrange)
import llama
import mistral
import baichuan
import pickle
import argparse
from validation.train_locators import Locator
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy import integrate
from joblib import Parallel, delayed
from torch.distributions import MultivariateNormal

HF_NAMES = {
    'llama_7B': '/path/to/your/workdir/huggingface/decapoda-research-llama-7B-hf',
    'alpaca_7B': '/path/to/your/workdir/huggingface/alpaca-7b', 
    'vicuna_7B': '/path/to/your/workdir/huggingface/vicuna-7b', 
    'llama2_chat_7B': '/path/to/your/workdir/huggingface/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': '/path/to/your/workdir/huggingface/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': '/path/to/your/workdir/huggingface/Llama-2-70b-chat-hf', 
    'llama3_instruct_8B': '/path/to/your/workdir/huggingface/llama3-instruct-8b',
    'mistral_7B': '/path/to/your/workdir/huggingface/mistral-7b',
    'baichuan_chat_7B': '/path/to/your/workdir/huggingface/baichuan-7b-chat',
}
def standardize(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / std

def calculate_entropy_via_histogram(tensor, num_bins=100):

    bin_edges = torch.linspace(0, 1, num_bins + 1, device=tensor.device)
    bin_indices = torch.bucketize(tensor, bin_edges) - 1  
    # print(bin_indices.min(), bin_indices.max())
    bin_indices = bin_indices.clamp(min=0, max=num_bins-1)
    bin_counts = torch.bincount(bin_indices, minlength=num_bins).float()
    
    probs = bin_counts / bin_counts.sum()
    
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10)) 

    return entropy.item()

def calculate_joint_entropy_via_histogram(tensor_x, tensor_y, num_bins=100):
    
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=tensor_x.device)
    bin_indices_x = torch.bucketize(tensor_x, bin_edges) - 1
    bin_indices_x = bin_indices_x.clamp(min=0, max=num_bins-1)
    bin_indices_y = torch.bucketize(tensor_y, bin_edges) - 1
    bin_indices_y = bin_indices_y.clamp(min=0, max=num_bins-1)
    
    joint_indices = bin_indices_x * num_bins + bin_indices_y
    joint_counts = torch.bincount(joint_indices, minlength=num_bins*num_bins).float()
    
    joint_probs = joint_counts / joint_counts.sum()
    
    joint_entropy = -torch.sum(joint_probs * torch.log2(joint_probs + 1e-10))  

    return joint_entropy.item()

def mutual_information(tensor_x, tensor_y, num_bins=100):
    # tensor_x = torch.softmax(tensor_x.squeeze(), dim=0)
    tensor_x = (tensor_x - tensor_x.min()) / (tensor_x.max() - tensor_x.min())
    tensor_y = (tensor_y - tensor_y.min()) / (tensor_y.max() - tensor_y.min())
    # tensor_y = torch.softmax(tensor_y.squeeze(), dim=0)
    H_X = calculate_entropy_via_histogram(tensor_x.squeeze(), num_bins)
    H_Y = calculate_entropy_via_histogram(tensor_y.squeeze(), num_bins)
    H_XY = calculate_joint_entropy_via_histogram(tensor_x.squeeze(), tensor_y.squeeze(), num_bins)
    
    mutual_information = H_X + H_Y - H_XY
    return mutual_information


def compute_pairwise_mutual_information(data, bandwidth=1.0):
    # data = standardize(data)
    n_samples = data.shape[0]
    mi_matrix = torch.zeros((n_samples, n_samples), device=data.device)
    for i in range(n_samples):
        for j in range(i, n_samples):
            mi_matrix[i, j] = mutual_information(data[i].unsqueeze(0), data[j].unsqueeze(0))
            mi_matrix[j, i] = mi_matrix[i, j]
    return mi_matrix

def get_graph_mi_feature(head_wise_activations, start, t=1):
    layer_gcn_avg = [[] for _ in range(t)]
    for layer in range(head_wise_activations.shape[0]):
        answers = head_wise_activations[layer, start:, :]
        for _ in range(t):
            mi_matrix = compute_pairwise_mutual_information(torch.tensor(answers).cuda())
            # norm = answers / np.linalg.norm(answers, axis=1, keepdims=True)
            # matrix = np.dot(norm, norm.T)
            exp_matrix = np.exp(mi_matrix.cpu().numpy())
            softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

            answers = np.dot(softmax_matrix, answers)
            layer_gcn_avg[_].append(np.average(answers, axis=0, keepdims=True))
        # layer_avg = np.average(answers, axis=0, keepdims=True)
        # answers = head_wise_activations[min_idx - 1:, :]
        # layer_avg_gt = np.average(answers[min_idx - 1:, :], axis=0, keepdims=True)
        # layer_gcn_avg.append(layer_avg)
        # layer_gcn_avggt.append(layer_avg_gt)
    return layer_gcn_avg
def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_instruct_8B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # tokenizer = llama.LlamaTokenizerFast.from_pretrained(MODEL)
    # model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    ## tokenizer
    if 'baichuan' in args.model_name:
        tokenizer = baichuan.BaichuanTokenizer.from_pretrained(MODEL)
    elif 'llama3' in args.model_name: 
        tokenizer = llama.LlamaTokenizerFast.from_pretrained(MODEL)
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    
    
    if 'llama' in args.model_name or 'vicuna' in args.model_name or 'alpaca' in args.model_name:
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    elif 'baichuan' in args.model_name:
        model = baichuan.BaichuanForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    elif 'mistral' in args.model_name:
        model = mistral.MistralForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    else:
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")

    device = "cuda"

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("/path/to/your/workdir/tae/truthful_qa_hf", "multiple_choice")['validation']
        # formatter = tokenized_tqa
        formatter = tokenized_tqa_with_start
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("/path/to/your/workdir/tae/truthful_qa_hf", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("/path/to/your/workdir/tae/truthful_qa_hf", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'nq':
        dataset = load_nq()
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels, starts = formatter(dataset, tokenizer)
    if args.dataset_name == 'tqa_mc2':    
        key_words, answers = get_critical_words(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []
    
    ### val probe
    all_head_wise_activations_total = []
    all_labels = []
    all_lens = []

    ## train locator
    locator_activations_neg = []
    locator_starts_neg = []
    locator_labels_neg = []

    ### mi feature
    k = 1
    all_head_wise_activations_average_gcnmi = [[] for _ in range(k)]

    print("Getting activations")
    if args.dataset_name == 'tqa_mc2':
        for prompt, start, label, words in tqdm(zip(prompts, starts, labels, key_words)):
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
            
            ## MI activations
            layer_gcnmi_avg_all = get_graph_mi_feature(head_wise_activations, start, k)
            for _, layer_gcnmi_avg in enumerate(layer_gcnmi_avg_all):
                layer_gcnmi_avg = np.concatenate(layer_gcnmi_avg, axis=0)
                all_head_wise_activations_average_gcnmi[_].append(layer_gcnmi_avg)

            ### locator_activations neg
            if label < 1:
                locator_activations_neg.append(head_wise_activations)
                locator_starts_neg.append(start)
                loc_labels = [0 for _ in range(len(words['diff']) + len(words['same']))]

                assert len(loc_labels) == head_wise_activations[:, start: :].shape[-2]
                for d in words['diff']:
                    loc_labels[d-1] = 1
                locator_labels_neg.append(loc_labels)

            all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
            all_head_wise_activations.append(head_wise_activations[:,-1,:])
             
            ## all activations
            for j in range(start, head_wise_activations.shape[1]):
                all_labels.append(label)
                all_head_wise_activations_total.append(head_wise_activations[:,j,:])
            all_lens.append(head_wise_activations.shape[1]-start)

    else:
        for prompt, label in tqdm(zip(prompts, labels)):
            layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
            all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
            all_head_wise_activations.append(head_wise_activations[:,-1,:])
            
    print("Saving labels")
    np.save(f'features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)
    
    if args.dataset_name == 'tqa_mc2':
        for i, f in enumerate(all_head_wise_activations_average_gcnmi):
            print("Saving head wise activations average gcnmi " + str(i))
            np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise_average_gcnmi_{i}.npy', f)

        print("Saving locator_activations_neg")
        with open(f'features/{args.model_name}_{args.dataset_name}_locator_activations_neg.npy', 'wb') as f:
            pickle.dump(locator_activations_neg, f)
        
        print("Saving locator_starts_neg")
        np.save(f'features/{args.model_name}_{args.dataset_name}_locator_starts_neg.npy', locator_starts_neg)
        
        print("Saving locator_labels_neg")
        np.save(f'features/{args.model_name}_{args.dataset_name}_locator_labels_neg.npy', locator_labels_neg)

        
        print("Saving head wise activations total")
        np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise_total.npy', all_head_wise_activations_total)
        
        print("Saving labels all")
        np.save(f'features/{args.model_name}_{args.dataset_name}_labels_all.npy', all_labels)

        print("Saving lens all")
        np.save(f'features/{args.model_name}_{args.dataset_name}_lens.npy', np.array(all_lens))


if __name__ == '__main__':
    main()
