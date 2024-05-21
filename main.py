import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import copy
from distutils.util import strtobool


from utils import *
from model import *
from functions import *
import torch.optim as optim

def main(args):
  
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")


    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

        
    lang = Lang(train_raw, ["<pad>", "<eos>"])


    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)


    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(train_dataset, batch_size= args.batch_dimentions_Train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_dimentions_Val_Test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_dimentions_Val_Test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))



    # With SGD try with an higher learning rate (> 1 for instance)
    lr = 5 # This is definitely not good for SGD
    clip = 5 # Clip the gradient

    vocab_len = len(lang.word2id)

    model = LM_RNN( args ,  vocab_len, pad_index=lang.word2id["<pad>"]).to(args.device)
    model.apply(init_weights)
 

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    optimizer = setup_optimizer(model, args)
  
    patience = 10
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(1,args.N_epochs))

    metrcis_list = {}
    
    for epoch in pbar:

        metrcis_list[epoch] = {}

        loss = train_loop(args , train_loader, optimizer, criterion_train, model , metrcis_list[epoch])    
            

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            metrcis_list[epoch]["perplexity"] = ppl_dev
            metrcis_list[epoch]["loss_dev"] = loss_dev

            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            """  
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
            """
    
      
        if (epoch + 1) % 10 == 0:
            optimizer.update_model_parameters()
        
        optimizer.update_zero()
   
    
    best_model.to(args.device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    write_args_to_csv(args , Best_train_ppl = best_ppl , Test_ppl= final_ppl)
    plot_training_metrics(metrcis_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--device', type=str, default='cuda',
                        help="Specify the device to use for computation, e.g., 'cuda:0', 'cuda:1', or 'cpu'")
    parser.add_argument("--tokenizer_type", default="bert-base-uncased", type=str, help="Tokenizer type for bert" )

    parser.add_argument("--learning_rate", default="5", type=float, help="Learning rate for the model" )


    parser.add_argument("--batch_dimentions_Train", default="64", type=int, help="Dimentions of the Batch_size for the training " )
    parser.add_argument("--batch_dimentions_Val_Test", default="32", type=int, help="Dimentions of the Batch_size for the validation or Test set " )

    parser.add_argument("--N_epochs", default="100", type=int, help="Number of epochs" )
    parser.add_argument("--clip", default="5", type=int, help="Clip of the gradient" )

    parser.add_argument("--out_dropout", default="0.1", type=float, help="Dropout LL out model" )
    parser.add_argument("--emb_dropout", default="0.1", type=float, help="Dropout LL Embde" )

    parser.add_argument("--hidden_size", default="750", type=int, help="Hidden size of LSTM" )
    parser.add_argument("--embde_size", default="750", type=int, help="Embedding dimentions for lstm")

    parser.add_argument("--n_layers", default="1", type=int, help="Numbers of layer of LSTM")

    parser.add_argument("--Bi_directional", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Enable bi-directional LSTM. Use --Bi_directional=True to enable or --Bi_directional=False to disable.")
    
    parser.add_argument("--patience", default="10", type=int, help="Patient, stop if we not improving our model")
    
    #optimizer
    parser.add_argument("--optimizer_name", default="NTAvSGD", type=str, help="Types of optimizer")

    parser.add_argument("--momentum", default="0.4", type=float, help="Optimizer Momentum")
    parser.add_argument("--weight_decay", default="1e-4", type=float, help="Optimizer weight_decay")
    parser.add_argument("--trigger_threshold", default="5", type=int, help="Trigger for NTAvSGD")

    
    args = parser.parse_args()

    
   
    main(args)