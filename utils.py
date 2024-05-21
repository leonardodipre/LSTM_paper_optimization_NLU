import os
import csv
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
import torch.utils.data as data
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from functions import *

DEVICE = 'cuda' # it can be changed with 'cpu' if you do not have a gpu


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output



class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    
    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item



def str2bool(v):
    try:
        return bool(strtobool(v))
    except ValueError:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_optimizer(model, args):
    
    if "NTAvSGD" == args.optimizer_name:
        print("Optimizer selected NTAvSGD" )
        return NTAvSGD(model, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, trigger_threshold=args.trigger_threshold)
    
    elif "Adam" == args.optimizer_name:
        print("Optimizer selected Adam" )
        return Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    elif "SGD" == args.optimizer_name:
        print("Optimizer selected SGD" )
        return SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    else:
        raise ValueError("Unsupported optimizer")


def write_args_to_csv(args, Best_train_ppl  , Test_ppl , filepath="parames_our.csv"):
    
    args_dict = vars(args)
    
    args_dict['Best_train_ppl'] = Best_train_ppl
    args_dict['Perplexity Test'] = Test_ppl

    file_exists = os.path.isfile(filepath)
    

    with open(filepath, mode='a', newline='') as file:
        # Create a writer object from csv module
        writer = csv.DictWriter(file, fieldnames=args_dict.keys())
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(args_dict)


def plot_training_metrics(metrics, filename='training_metrics.png'):
    """
    Plots the training loss, perplexity, and development loss across epochs.

    Parameters:
    - metrics (dict): A dictionary where the key is the epoch number and the value is another
                      dictionary containing 'loss', 'perplexity', and 'loss_dev' for that epoch.
    - filename (str): The name of the file to save the plot.
    """
    # Convert the metrics dictionary into a DataFrame
    df = pd.DataFrame.from_dict(metrics, orient='index')

    # Create a figure with 3 subplots, one for each metric
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot the training loss
    df['loss'].plot(ax=axs[0], marker='o', linestyle='-', color='b', title='Training Loss vs Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    # Plot the training perplexity
    df['perplexity'].plot(ax=axs[1], marker='o', linestyle='-', color='r', title='Training Perplexity vs Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Perplexity')

    # Plot the development loss
    df['loss_dev'].plot(ax=axs[2], marker='o', linestyle='-', color='g', title='Dev Loss vs Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Dev Loss')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()