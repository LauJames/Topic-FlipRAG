import argparse
import bisect
import os
import json
import sys
from collections import defaultdict, Counter
import json
from torch.nn.functional import cosine_similarity
import torch
import tqdm
import numpy as np
from transformers import BertTokenizer, BertTokenizerFast, BertForNextSentencePrediction, AutoModelForSequenceClassification

from torch.autograd import Variable
from torch import cuda
import torch.nn.functional as F
from copy import deepcopy
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from apex import amp


from constraints_utils import create_constraints, get_sub_masks ,get_inputs_filter_ids ,STOPWORDS


curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = '/bert_ranker/results'  
prodir2 = "/adversarial_data/results"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  

BIRCH_DIR = prodir + '/data/birch'
BIRCH_MODEL_DIR = BIRCH_DIR + '/models'
BIRCH_DATA_DIR = BIRCH_DIR + '/data'
BIRCH_ALPHAS = [1.0, 0.5, 0.1]
BIRCH_GAMMA = 0.6
BERT_LM_MODEL_DIR = 'wiki103/bert/'  
BOS_TOKEN = '[unused0]'

device = 'cuda' if cuda.is_available() else 'cpu'
device_cpu = torch.device("cpu")

def main():
    parser = argparse.ArgumentParser('Collision_Attack')
    parser.add_argument('--mode', default='test', type=str,
                        help='train/test')
    parser.add_argument("--experiment_name", default='collision.pointwise', type=str)
    parser.add_argument("--target", type=str, default='nb_bert', help='test on what model')
    parser.add_argument("--target_type", type=str, default='none', help='target model of what kind of trigger')
    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--method", default="nature", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path", default="model/nbbert_embedding_adv.pt", type=str)
    parser.add_argument("--transformer_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", type=str, required=False, help="Bert model to use (cross-encoder/ms-marco-MiniLM-L-12-v2,bert-base-uncased).")
    parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
    parser.add_argument('--lr', type=float, default=0.005, help='optimization step size')
    parser.add_argument('--max_iter', type=int, default=30, help='maximum iteraiton')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
    parser.add_argument("--beta", default=0., type=float, help="Coefficient for language model loss.")
    parser.add_argument("--amount", default=0, type=int, help="adv_Data amount.")
    parser.add_argument('--save', action='store_true', help='Save collision to file')
    parser.add_argument('--verbose', action='store_true', default=True,  help='Print every iteration')
    parser.add_argument("--lm_model_dir", default=BERT_LM_MODEL_DIR, type=str, help="Path to pre-trained language model")
    parser.add_argument('--perturb_iter', type=int, default=50, help='PPLM iteration')
    parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
    parser.add_argument("--topk", default=10, type=int, help="Top k sampling for beam search")
    parser.add_argument("--num_beams", default=3, type=int, help="Number of beams")
    parser.add_argument("--num_filters", default=100, type=int, help="Number of num_filters words to be filtered")
    parser.add_argument('--nature', action='store_true', help='Nature collision')
    parser.add_argument('--pat', action='store_true', help='PAT.')
    parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')
    parser.add_argument('--fp16', default=True, action='store_true', help='fp16')
    parser.add_argument('--patience_limit', type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--seed", default=42, type=str, help="random seed")
    
    # python stage2_adversarial_trigger_generation copy.py.py --beta=0.0 --stemp=1.0 --num_beams=3 --topk=10 --max_iter=30 --mode=train
    args = parser.parse_args()


    tokenizer = BertTokenizerFast.from_pretrained('nboost:pt-bert-base-uncased-msmarco')
    model = AutoModelForSequenceClassification.from_pretrained('nboost:pt-bert-base-uncased-msmarco')
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
  
    model,  = amp.initialize([model])
    path='PROCON_data.json'
    with open(path, "r") as json_file:
        data = json.load(json_file)


    label_list = [0]
    for label in label_list:
        for i in range(0,42):
            if i==36 and label==1:
                continue
            example = data[i]
            query_list = example['queries'][:]
            topic = example['topic']
            query_list.append(topic)
            path_know='path_know'
            if not os.path.exists(path_know):
                continue
            trigger_list = []
            with open(path_know, "r") as json_file:
                data1 = json.load(json_file)
            for t in range(len(data1)):
                num_pas = data1[t]['num']
                target_passage = data1[t]['attack_passage']
                trigger, new_score, trigger_cands = gen_topic_adv_trigger(
                    inputs_a=query_list, 
                    inputs_b=None,  # best_sent 
                    model=model, 
                    tokenizer=tokenizer,
                    device=device, 
                    margin=None, 
                    lm_model=None, 
                    args=args,
                    target_passge=target_passage
                )
                print(trigger)
                trigger_dict = {
                    'num': num_pas,
                    'know_passage': target_passage,
                    'trigger': trigger
                }
                trigger_list.append(trigger_dict)
            print(trigger_list)
            with open(f"output_path/opinion_result_{i}_{label}_top5.json", "w") as json_file:
                json.dump(trigger_list, json_file, indent=4)

            

def find_filters(queries, model, tokenizer, device, k=100):
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    combined_scores = torch.zeros(len(words), device=device)
    for query in queries:
        inputs = tokenizer.batch_encode_plus([[query, w] for w in words], pad_to_max_length=True)
        all_input_ids = torch.tensor(inputs['input_ids'], device=device)
        all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
        all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
        n = len(words)
        batch_size = 512
        n_batches = n // batch_size + 1
        all_scores = []

        for i in tqdm(range(n_batches), desc='Processing queries'):
            input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
            token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
            attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
            outputs = model.forward(input_ids, attention_masks, token_type_ids)
            scores = outputs[0][:, 1]
            all_scores.append(scores)
        all_scores = torch.cat(all_scores)
        combined_scores += all_scores
    _, top_indices = torch.topk(combined_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]

def find_filters_anchor(queries, anchor_list, model, tokenizer, device, k=150):
    combined_anchor = ' '.join(anchor_list)
    words = list(set(combined_anchor.split()))
    words = [w for w in words if w.isalpha()]
    combined_scores = torch.zeros(len(words), device=device)
    for query in queries:
        pairs = [[query, w] for w in words]
        inputs = tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            return_tensors='pt'
        )
    
        input_ids = inputs['input_ids'].to(device)
        attention_masks = inputs['attention_mask'].to(device)
        token_type_ids = inputs.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        n = len(words)
        batch_size = 512
        n_batches = (n + batch_size - 1) // batch_size
        all_scores = []
        
        for i in tqdm(range(n_batches), desc='Processing queries'):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch_input_ids = input_ids[batch_slice]
            batch_attention_masks = attention_masks[batch_slice]
            batch_token_type_ids = token_type_ids[batch_slice] if token_type_ids is not None else None
        
            with torch.no_grad():
                if batch_token_type_ids is not None:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_masks,
                        token_type_ids=batch_token_type_ids
                    )
                else:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_masks
                    )
            logits = outputs.logits
            scores = logits[:, 1]  
            all_scores.append(scores)
        all_scores = torch.cat(all_scores)
        combined_scores += all_scores
    _, top_indices = torch.topk(combined_scores, k)
    filters = [words[i] for i in top_indices.tolist()]
    return filters


def find_top_relevant_tokens(queries, model, tokenizer, device, k=150):
  
    words = [w for w in tokenizer.vocab.keys() if w.isalpha() and w.lower() not in STOPWORDS]
    word_ids = tokenizer.convert_tokens_to_ids(words)
    word_ids_tensor = torch.tensor(word_ids, device=device)
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        word_embeddings = embedding_layer(word_ids_tensor)  
    combined_scores = torch.zeros(len(words), device=device)
    for query in tqdm(queries, desc='Processing queries'):
        query_inputs = tokenizer(query, return_tensors='pt').to(device)
        input_ids = query_inputs['input_ids']  
        with torch.no_grad():
            
            query_embedding = embedding_layer(input_ids).mean(dim=1) 
        similarities = cosine_similarity(query_embedding, word_embeddings) 
        combined_scores += similarities.squeeze()
    
    top_scores, top_indices = torch.topk(combined_scores, k)
    top_tokens = [words[i] for i in top_indices.cpu().numpy()]
    
    return top_tokens

def get_queries_tokens(queries, tokenizer, top_n=30):
    tokens_in_queries = defaultdict(set)  
    for idx, text in enumerate(queries):
        query_tokens = tokenizer.tokenize(text)
        filtered_tokens = [t.lower() for t in query_tokens if t.isalpha() and len(t) > 2 and t.lower() not in STOPWORDS]
        for token in filtered_tokens:
            tokens_in_queries[token].add(idx)

    token_freq = {token: len(query_indices) for token, query_indices in tokens_in_queries.items()}
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [token for token, freq in sorted_tokens[:top_n]]
    return top_tokens

def gen_topic_adv_trigger(inputs_a, inputs_b, model, tokenizer, device, margin=None, lm_model=None, args=None, target_passge=None):
    def relaxed_to_word_embs(x):
        masked_x = x + input_mask + sub_mask
        p = torch.softmax(masked_x / args.stemp, -1)
        x = torch.mm(p, word_embedding)
        return p, x.unsqueeze(0)

    def ids_to_emb(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb

    def ids_to_emb_passage(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb

    word_embedding = model.get_input_embeddings().weight.detach()
    vocab_size = word_embedding.size(0)
    input_mask = torch.zeros(vocab_size, device=device)
    filters = find_top_relevant_tokens(inputs_a, model, tokenizer, device, k=args.num_filters)
    input_mask[tokenizer.convert_tokens_to_ids(filters)] = 0.68
    stopwords_mask = create_constraints(args.seq_len, tokenizer, device)

    best_score = -1e9
    z_i = torch.zeros((args.seq_len, vocab_size), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([z_i], lr=args.lr)

    for it in range(args.max_iter):
        for j in range(args.perturb_iter):
            input_embeds_list, attention_mask_list = [], []
            for query_id in inputs_a:
                query_emb = ids_to_emb(query_id.unsqueeze(0))
                p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
                concat_embeds = torch.cat([query_emb, inputs_embeds], dim=1)
                input_embeds_list.append(concat_embeds)
                attention_mask_list.append(torch.ones(concat_embeds.shape[1], dtype=torch.long, device=device))

            optimizer.zero_grad()
            outputs = model(inputs_embeds=torch.cat(input_embeds_list), attention_mask=torch.cat(attention_mask_list))
            loss, logits = outputs[0], outputs[1]
            loss.backward()
            optimizer.step()

        avg_score = logits[:, 1].mean().item()
        if avg_score > best_score:
            best_score = avg_score
            best_collision = tokenizer.decode(torch.argmax(z_i, dim=-1).cpu().tolist())
        else:
            break

    return best_collision, best_score

if __name__ == '__main__':
    main()