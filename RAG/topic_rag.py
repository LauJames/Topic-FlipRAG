from langchain_community.vectorstores import FAISS
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
from tqdm import tqdm

import torch
from LocalEmbedding import (
    localEmbedding_sentence_dpr,
    localEmbedding_sentence_ance,
    localEmbedding_sentence_contriever
)
from ConversationRAG import ConversationRAGChain
from evaluate import (
    cal_NDCG,
    topk_proportion,
    topk_mutual_score,
    relabel_polarity
)


import json
from langchain.vectorstores.faiss import FAISS
from qwen_LLM import Qwen_LLM
from llama3 import LLaMA3_LLM
from llama3_1 import LLAMA3_1_LLM
import random
from langchain.prompts import ChatPromptTemplate
from rag_utils import extract_by_symbol


def load_data(label):
    path = 'PROCON_data.json'
    with open(path, "r",encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    target_query = []
    texts = []
    texts_attacked = []
    text_label_dict = {}
    att_label_dict = {}
    target_category=[]
    poisoned_num_list=[]
    topic_list=[]

    for i in range(0,42):
        path_1 = f'10.30_baseline_result/opinion_result_{i}_{label}.json'
        if not os.path.exists(path_1):
            continue 
        with open(path_1, 'r',encoding='utf-8') as f:
            result = json.load(f)
        passage_ori = [item['passage_ori'] for item in result][:5]
        passage_know = [item['know_passage'] for item in result][:5]
        trigger = [item['trigger'] for item in result]
        
        example = data[i]
        query_list = example['queries']
        topic=example['topic']
        topic_list.append(topic)
        target_query.append(query_list)
        category=example['category']
        target_category.append(category)
        label_list = [t[1] for t in example['passages']]
        passages = [t[3] for t in example['passages']]
        passages_ori = passages.copy()
        texts.extend(passages_ori)
        passages_know = passages.copy()
        passages_final = passages.copy()
        
        for idx, passage in enumerate(passages):
            if passage in passage_ori:
                ori_index = passage_ori.index(passage)
                passages_know[idx] = passage_know[ori_index]
                passages_final[idx] = trigger[ori_index] + ' ' + passage_know[ori_index]
             
        texts_attacked.extend(passages_final)
        for passage, lbl in zip(passages, label_list):
            text_label_dict[passage] = lbl  
        for passage, lbl in zip(passages_final, label_list):
            att_label_dict[passage] = lbl

    return target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list

path = 'PROCON_data'
def load_data_ablation(label):

    with open(path, "r",encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    target_query = []
    texts = []
    texts_attacked = []
    text_label_dict = {}
    att_label_dict = {}
    target_category=[]
    poisoned_num_list=[]
    topic_list=[]

    for i in range(0,42):

        path_1 = f'Topic_FlipRAG_result_{i}_{label}.json'
        if not os.path.exists(path_1):
            continue 

        with open(path_1, 'r',encoding='utf-8') as f:
            result = json.load(f)
   
        passage_ori = [item['passage_ori'] for item in result][:5]
    
        passage_know=[item['attack_passage'] for item in result][:5]
        trigger = [item['trigger'] for item in result]
        example = data[i]
        category=example['category']
    
        query_list = example['queries']
        topic=example['topic']
        topic_list.append(topic)
        target_query.append(query_list)

        target_category.append(category)
        label_list = [t[1] for t in example['passages']]
        passages = [t[3] for t in example['passages']]
        passages_ori = passages.copy()
        texts.extend(passages_ori)
        passages_know = passages.copy()
        passages_final = passages.copy()
        for idx, passage in enumerate(passages):
            if passage in passage_ori:
                ori_index = passage_ori.index(passage)
                passages_know[idx] = passage_know[ori_index]
                passages_final[idx] = trigger[ori_index] + ' ' + passage_know[ori_index]
           
        texts_attacked.extend(passages_final)
        for passage, lbl in zip(passages, label_list):
            text_label_dict[passage] = lbl  
        for passage, lbl in zip(passages_final, label_list):
            att_label_dict[passage] = lbl
    print(len(target_category))
    return target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list


def load_data_query(label):
    with open(path, "r",encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    target_query = []
    texts = []
    texts_attacked = []
    text_label_dict = {}
    att_label_dict = {}
    target_category=[]
    poisoned_num_list=[]
    topic_list=[]

    for i in range(0,42):

        path_1 = f'Topic_FlipRAG_result_{i}_{label}.json'
        if not os.path.exists(path_1):
            continue 
        with open(path_1, 'r',encoding='utf-8') as f:
            result = json.load(f)
        passage_ori = [item['passage_ori'] for item in result][:5]
        example = data[i]
        query_list = example['queries']
        trigger = query_list[3]
        topic=example['topic']
        topic_list.append(topic)
        target_query.append(query_list)
        category=example['category']
        target_category.append(category)
        label_list = [t[1] for t in example['passages']]
        passages = [t[3] for t in example['passages']]
        passages_ori = passages.copy()
        texts.extend(passages_ori)
        passages_final = passages.copy()
        for idx, passage in enumerate(passages):
            if passage in passage_ori:
                ori_index = passage_ori.index(passage)
                passages_final[idx] = trigger + ' ' + passage_ori[ori_index]
        texts_attacked.extend(passages_final)
        for passage, lbl in zip(passages, label_list):
            text_label_dict[passage] = lbl  
        for passage, lbl in zip(passages_final, label_list):
            att_label_dict[passage] = lbl

    return target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list

def load_data_collision(label):
    with open(path, "r",encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    target_query = []
    texts = []
    texts_attacked = []
    text_label_dict = {}
    att_label_dict = {}
    target_category=[]
    topic_list=[]

    for i in range(0,42):
        path_collision='/opinion_collision_triggers_group_wo_agu.json'
        path_1 = f'/result_topic/10.30_baseline_result/opinion_result_{i}_{label}.json'
        if not os.path.exists(path_1):
            continue 
        with open(path_1, 'r',encoding='utf-8') as f:
            result = json.load(f)
        with open(path_collision, 'r',encoding='utf-8') as f1:
            triggers_collision=json.load(f1)
        passage_ori = [item['passage_ori'] for item in result][:5]
     
        trigger = triggers_collision[i]
        example = data[i]
        query_list = example['queries']
        topic=example['topic']
        topic_list.append(topic)
        target_query.append(query_list)
        category=example['category']
        target_category.append(category)
        label_list = [t[1] for t in example['passages']]
        passages = [t[3] for t in example['passages']]
        passages_ori = passages.copy()
        texts.extend(passages_ori)

        passages_final = passages.copy()
        for idx, passage in enumerate(passages):
            if passage in passage_ori:
                ori_index = passage_ori.index(passage)
   
                passages_final[idx] = trigger + ' ' + passage_ori[ori_index]
        texts_attacked.extend(passages_final)
        for passage, lbl in zip(passages, label_list):
            text_label_dict[passage] = lbl  
        for passage, lbl in zip(passages_final, label_list):
            att_label_dict[passage] = lbl

    return target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list

def load_data_pat(label):
    with open(path, "r",encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    target_query = []
    texts = []
    texts_attacked = []
    text_label_dict = {}
    att_label_dict = {}
    target_category=[]
    poisoned_num_list=[]
    topic_list=[]

    for i in range(0,42):
        path_1=f'opinion_result_{1}_{label}.json'
        path_2 = f'pat_topic_opinion_triggers_{i}_{label}.json'
        if not os.path.exists(path_1):
            print(1)
            continue 
        if not os.path.exists(path_2):
            print(2)
            continue 
        with open(path_1, 'r',encoding='utf-8') as f:
            result = json.load(f)
        with open(path_2,'r',encoding='utf-8') as f:
            result_pat = json.load(f)
        passage_ori = [item['ori_passage'] for item in result_pat][:5]
        trigger=[item['trigger'] for item in result_pat]
        #use_or_not=[item['use_or_not'] for item in result]
        
        example = data[i]
        query_list = example['queries']
        topic=example['topic']
        topic_list.append(topic)
        target_query.append(query_list)
        category=example['category']
        target_category.append(category)
        label_list = [t[1] for t in example['passages']]
        passages = [t[3] for t in example['passages']]
        passages_ori = passages.copy()
        texts.extend(passages_ori)
        passages_know = passages.copy()
        passages_final = passages.copy()
        
        #poisoned_num=0
        for idx, passage in enumerate(passages):
            if passage in passage_ori:
                ori_index = passage_ori.index(passage)
                passages_final[idx] = trigger[ori_index] + ' ' + passage_ori[ori_index]
        texts_attacked.extend(passages_final)
        for passage, lbl in zip(passages, label_list):
            text_label_dict[passage] = lbl  
        for passage, lbl in zip(passages_final, label_list):
            att_label_dict[passage] = lbl

    return target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list


def create_embeddings(texts, embedding_model):
    db =  FAISS.from_texts(texts, embedding_model, distance_strategy = 'MAX_INNER_PRODUCT')
    return db

def create_retriever(db, topk):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": topk})
    return retriever

def load_llm(model_path, device='cuda'):
    llm = Qwen_LLM(mode_name_or_path=model_path, device=device)
    llm.eval()
    return llm

def evaluate_retrieval(serach_result, serach_result_attacked, text_label_dict, att_label_dict, target_polarity, topk):
    serach_result_label = []
    serach_result_attacked_label = []
    for i in range(len(serach_result)):
        serach_result_label.append(text_label_dict.get(serach_result[i].page_content, 1 - target_polarity))
        serach_result_attacked_label.append(att_label_dict.get(serach_result_attacked[i].page_content, 1 - target_polarity))
    return serach_result_label, serach_result_attacked_label

def rag_generation(label,rag_type='conversation',llm='qwen',dr='dpr',topk = 3,attack_type='know'):
  
    target_polarity = label
    device = 'cuda'
    if dr=='dpr':
        CON_NAME = '/DR/dpr'
        embedding_model = localEmbedding_sentence_dpr(CON_NAME, device)
    elif dr=='ance':
        CON_NAME = '/DR/ance'
        embedding_model = localEmbedding_sentence_ance(CON_NAME, device)
    elif dr=='contriever':
        CON_NAME = 'DR/contriever'
        embedding_model = localEmbedding_sentence_contriever(CON_NAME, device)
    if attack_type=='know':
        target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list = load_data(label)
    elif attack_type=='query+':
        target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list = load_data_query(label)
    elif attack_type=='collision':
        target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list = load_data_collision(label)
    elif attack_type=='pat':
        target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list = load_data_pat(label)
    elif attack_type=='ablation':
        target_query, texts, texts_attacked, text_label_dict, att_label_dict,target_category,topic_list = load_data_ablation(label)
  

    db = create_embeddings(texts, embedding_model)
    db_attacked = create_embeddings(texts_attacked, embedding_model)

    retriever = create_retriever(db, topk)
    retriever_attacked = create_retriever(db_attacked, topk)
    name_llm=llm

    if llm=='qwen':
        model_path_qwen14 = "Qwen2.5-7B-Instruct-AWQ"
        llm = load_llm(model_path_qwen14,device=device)

    elif llm=='llama3.1':
        model_path_llama31 = 'Meta-Llama-3.1-8B-Instruct'
        llm = LLAMA3_1_LLM(model_name_or_path=model_path_llama31)
        
    
    if rag_type ==  "conversation":
        rag_chain = ConversationRAGChain(llm=llm, prompt_llm = llm, retriever=retriever)
        rag_chain_attacked = ConversationRAGChain(llm=llm, prompt_llm = llm, retriever=retriever_attacked)
        results_data = []
        topk_3_variation_list = []
        for t in tqdm(range(len(target_query)), desc="Processing Topics"):
            all_ndcg_ori = []
            all_ndcg_atk = []
            sum_boost_list = []
            all_amount = 0
            ndcg_didder_list = []
            topk_ori_list = []
            topk_atk_list = []
            query_list = target_query[t]
            topic_cate = target_category[t]
            topic_ = topic_list[t]
       
            topic_data = {
                "topic":topic_,
                "topic_category": topic_cate,
                "queries": []
            }

            for idx, query in enumerate(query_list):
                print("QUERY:", query)
                serach_result = db.similarity_search(query, k=topk)
                serach_result_attacked = db_attacked.similarity_search(query, k=topk)

                serach_result_label, serach_result_attacked_label = evaluate_retrieval(
                    serach_result, serach_result_attacked, text_label_dict, att_label_dict, target_polarity, topk
                )

                print("SEARCH LABEL:", serach_result_label)
                print("ATTACK:", serach_result_attacked_label)
                result = topk_proportion(serach_result_label, serach_result_attacked_label, target_polarity, [topk])
                result.update(topk_mutual_score(serach_result_label, serach_result_attacked_label, target_polarity, [1/2]))
                for t in result:
                    print(t, ":", result[t])
                topk_ori_list.append(result['before-top'+str(topk)])
                topk_atk_list.append(result['later-top'+str(topk)])

                topk_3_result = topk_proportion(serach_result_label, serach_result_attacked_label, target_polarity, [3])
                topk_3_variation = topk_3_result['later-top3'] - topk_3_result['before-top3']
                topk_3_variation_list.append(topk_3_variation)

                pseudo_rank = list(range(len(serach_result_label)))
                pseudo_rank.reverse()
                pseudo_rank_atk = list(range(len(serach_result_attacked_label)))
                pseudo_rank_atk.reverse()
                ndcg_ori = cal_NDCG(pseudo_rank, relabel_polarity(target_polarity, serach_result_label), k=topk)
                ndcg_atk = cal_NDCG(pseudo_rank_atk, relabel_polarity(target_polarity, serach_result_attacked_label), k=topk)
                print("origianl NDCG:", ndcg_ori)
                print("After manipulation NDCG:", ndcg_atk)
                all_ndcg_ori.append(ndcg_ori)
                all_ndcg_atk.append(ndcg_atk)
                ndcg_differ = (ndcg_atk - ndcg_ori)
                ndcg_didder_list.append(ndcg_differ)

                if rag_type == "conversation":
                    out_origin = rag_chain.run("Here is the user question: <<<"+query+">>> \n Here is the command:DO NOT change a word")
                    out_atk = rag_chain_attacked.run("Here is the user question: <<<"+query+">>> \n Here is the command:DO NOT change a word")
                    
                    print(extract_by_symbol(out_origin['answer'], symbol="[[ ]]")[0])
            
                    print(extract_by_symbol(out_atk['answer'], symbol="[[ ]]")[0])
                   
                else:
                    out_origin = rag_chain.invoke(query)
                    out_atk = rag_chain_attacked.invoke(query)

                query_result = {
                    "query": query,
                    "output_origin": extract_by_symbol(out_origin['answer'], symbol="[[ ]]")[0] if isinstance(out_origin, dict) else out_origin,
                    "output_attacked": extract_by_symbol(out_atk['answer'], symbol="[[ ]]")[0] if isinstance(out_atk, dict) else out_atk,
                    "topk_origin_results": [doc.page_content for doc in serach_result],
                    "topk_attacked_results": [doc.page_content for doc in serach_result_attacked]
                }
                topic_data["queries"].append(query_result)

            topic_data["evaluation"] = {
                "Top3_origin": sum(topk_ori_list) / len(topk_ori_list) if topk_ori_list else 0,
                "Top3_attacked": sum(topk_atk_list) / len(topk_atk_list) if topk_atk_list else 0,
                "NDCG_ORI": sum(all_ndcg_ori) / len(all_ndcg_ori) if all_ndcg_ori else 0,
                "NDCG_ATK": sum(all_ndcg_atk) / len(all_ndcg_atk) if all_ndcg_atk else 0,
                "NDCG_variation": sum(ndcg_didder_list) / len(ndcg_didder_list) if ndcg_didder_list else 0
                
            }

            results_data.append(topic_data)

        with open(f'/output_baseline/llama3.1/{dr}_{name_llm}_{label}_{attack_type}_wo_dynamic_t3_a.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)



    if rag_type == 'direct':
        rag_chain = ConversationRAGChain(llm=llm, prompt_llm=llm, retriever=retriever)
        results_data = []
        chosen_topic=['Is Artificial Intelligence Good for Society?','Is Binge-Watching Good for You?','Should the Death Penalty Be Legal?','Should the United States Continue Its Use of Drone Strikes Abroad?','Should Public College Be Tuition-Free?']
        for t in tqdm(range(len(target_query)), desc="Processing Topics"):
            query_list = target_query[t]
            topic_cate = target_category[t]
            topic_ = topic_list[t]
            if topic_ not in chosen_topic:
                continue

            topic_data = {
                "topic_category": topic_cate,
                "topic": topic_,
                "queries": []
            }

            for idx, query in enumerate(query_list):
                out_direct = rag_chain.direct_answer(f"Here is the user question: <<<{query}>>> \n Here is the command:DO NOT change a word")
                print(query)
                print(out_direct)
                query_result = {
                    "query": query,
                    "output_direct": out_direct
                }
                topic_data["queries"].append(query_result)
            
            results_data.append(topic_data)

        with open(f'/direct_contriever_llama3.1.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)
      


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    rag_generation(label=0,rag_type='direct',llm='llama3.1',dr='contriever',attack_type='know')







