from openai import OpenAI
import torch
import numpy as np
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

client = OpenAI(api_key='your API key')
def gpt(a):
  prompt=a
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": prompt

      }
    ],
    temperature=1,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0
    )
  generated_text = response.choices[0].message

  return generated_text.content

model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def get_sim(passage1,passage2):
  embeddings1 = model_sim.encode(passage1, convert_to_tensor=True)
  embeddings2 = model_sim.encode(passage2, convert_to_tensor=True)
  cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
  return cosine_similarity.item()



tokenizer = BertTokenizerFast.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False
max_length = 512
def get_scores_nbbert(query_list,passage):
    query_passage_pairs = [(query, passage) for query in query_list]
    scores = []
    batch_size = 2  
    for i in range(0, len(query_passage_pairs), batch_size):
        batch_pairs = query_passage_pairs[i:i + batch_size]
        batch_encoding = tokenizer(batch_pairs,
                                max_length=max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')

        input_ids = batch_encoding['input_ids'].to(device)
        token_type_ids = batch_encoding['token_type_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        logits = outputs.logits  
        batch_scores = logits[:, 1].cpu().numpy().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]


        scores.extend(batch_scores)
    average_score = np.mean(scores)
    print(f"Average Score: {average_score}")
    return average_score


def get_tokens_num(input_string):
  tokens = tokenizer.encode(input_string, add_special_tokens=True)
  num_tokens = len(tokens)
  return num_tokens

def cac_num_pre(target_passage,modified_passage):
    original_tokens = len(target_passage.split())
    modified_tokens = len(modified_passage.split())
    token_difference = abs(modified_tokens - original_tokens)
    token_difference_percentage = token_difference / original_tokens
    return token_difference_percentage


def filter_passage(passage_list, target_passage,rel_limit=0.85, tokens_limit=0.2):
    filter_list = []
    #tokens_num_ori = get_tokens_num(target_passage)
    for item in passage_list:
        precentage = cac_num_pre(target_passage, item)
        rel_scores = get_sim(item, target_passage)
        if  precentage <= tokens_limit and rel_scores>=rel_limit:
            filter_list.append(item)
    if not filter_list:
        return None

    return filter_list

def find_best_passage(filter_list, target_passage, query_list):
    if not filter_list:
        return None

    scores_list = []
    for item in filter_list:
        score = get_scores_nbbert(query_list, item)
        scores_list.append((item, score))  

    best_item = max(scores_list, key=lambda x: x[1])  
    return best_item[0]

def sort_passages(query_list, passage_list):
    passage_scores = []
    for passage_text, passage_id in passage_list:
        score = get_scores_nbbert(query_list, passage_text)
        if get_tokens_num(passage_text) > 30:
            passage_scores.append((score, passage_text, passage_id))

    sorted_passages = sorted(passage_scores, key=lambda x: x[0], reverse=False)
    return sorted_passages