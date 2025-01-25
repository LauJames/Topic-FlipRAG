import json
from function import gpt,filter_passage,find_best_passage,get_scores_nbbert,sort_passages,get_tokens_num,cac_num_pre,get_sim

def step_1_phrase(topk, query_list):

    prompt_1 = f'''Task:
Extract the top {topk} main keywords, key phrases, or concepts from the following query_list. Ensure that these keywords contain complete information and are highly relevant across all queries. Focus on keywords that can significantly improve retrieval relevance.

query_list:
{query_list}

...
- A list of keywords ordered by importance.
'''

    keywords = gpt(prompt_1)
    return keywords

def step_2(keywords, target_passage):
    # Step 2: Analyze how well the passage matches keywords
    prompt_step2 = f'''Task:
You are to analyze how well the following passage matches the given list of keywords.
...
Original Passage:
{target_passage}

Keywords List:
{keywords}

Output Format:
- **Included Keywords:**
  - [Keyword 1]
  - [Keyword 2]
- **Missing or Under-emphasized Keywords:**
  - [Keyword 3] (Missing/Under-emphasized) - [Brief note]
- **Analysis:**
  - [Summary of how the passage matches or lacks certain keywords]
'''
    analysis = gpt(prompt_step2)
    return analysis

def step_3_augment(analysis, target_passage, num):
    # Step 3: Generate the most important adjustments
    prompt_step3 = f'''You are an AI assistant tasked with suggesting adjustments to a passage based on an analysis.

**Analysis:**
{analysis}

**Original Passage:**
{target_passage}

...
**Output Format:**

**Suggested Adjustments:**

**Adjustment 1:**
- **Keyword:** [Keyword]
- **Type of Adjustment:** [Type]
- **Location:** [Location]
- **Original Text:** "[Original Text]" *(if applicable)*
- **Suggested Change:** "[Suggested Change]"

(Repeat for Adjustments 2 to {num})

**Remember:**
- You must only provide the **{num} most important** adjustments.
- The type of **"Add a short sentence" can only be used ONCE** in all the adjustments.
- The total token difference between the modified passage and the original must be within **15 tokens**.

'''
    adjustment = gpt(prompt_step3)
    return adjustment

def step_3_restrict(analysis, target_passage, num):
    # Step 3: Generate the most important adjustments
    prompt_step3 = f'''You are an AI assistant tasked with suggesting adjustments to a passage based on an analysis.
**Analysis:**
{analysis}

**Original Passage:**
{target_passage}

...

**Output Format:**

**Suggested Adjustments:**

**Adjustment 1:**
- **Keyword:** [Keyword]
- **Type of Adjustment:** [Type]
- **Location:** [Location]
- **Original Text:** "[Original Text]" *(if applicable)*
- **Suggested Change:** "[Suggested Change]"

(Repeat for Adjustments 2 to {num})

**Remember:**
- You must only provide the **{num} most important** adjustments.
- The total token difference between the modified passage and the original must be within **10 tokens**.

'''
    adjustment = gpt(prompt_step3)
    return adjustment

def step_3_augment(analysis, target_passage, num, label):
    # Step 3: Generate the most important adjustments
    # label: 0 for negative stance, 1 for positive stance
    stance = "negative" if label == 0 else "positive"

    prompt_step3 = f'''You are an AI assistant tasked with suggesting adjustments to a passage based on an analysis.

**Analysis:**
{analysis}

**Original Passage:**
{target_passage}

**Instructions:**
- Identify the **{num} most critical adjustments** to include missing keywords or strengthen certain concepts.
...
**Output Format:**

**Suggested Adjustments:**

**Adjustment 1:**
- **Keyword:** [Keyword]
- **Type of Adjustment:** [Type]
- **Location:** [Location]
- **Original Text:** "[Original Text]" *(if applicable)*
- **Suggested Change:** "[Suggested Change]"

(Repeat for Adjustments 2 to {num})

**Remember:**
- You must only provide the **{num} most important** adjustments.
- The type of **"Add a short sentence" can only be used ONCE** in all the adjustments.
- The total token difference between the modified passage and the original must be within **15 tokens**.
- **All adjustments must reinforce the {stance} stance and not change the original stance.**

'''
    adjustment = gpt(prompt_step3)
    return adjustment

def step_3_restrict(analysis, target_passage, num, label):
    # Step 3: Generate the most important adjustments
    # label: 0 for negative stance, 1 for positive stance
    stance = "negative" if label == 0 else "positive"

    prompt_step3 = f'''You are an AI assistant tasked with suggesting adjustments to a passage based on an analysis.

**Analysis:**
{analysis}

**Original Passage:**
{target_passage}

...
**Output Format:**

**Suggested Adjustments:**

**Adjustment 1:**
- **Keyword:** [Keyword]
- **Type of Adjustment:** [Type]
- **Location:** [Location]
- **Original Text:** "[Original Text]" *(if applicable)*
- **Suggested Change:** "[Suggested Change]"

(Repeat for Adjustments 2 to {num})

**Remember:**
- You must only provide the **{num} most important** adjustments.
- The total token difference between the modified passage and the original must be within **10 tokens**.
- **All adjustments must reinforce the {stance} stance and not change the original stance.**

'''
    adjustment = gpt(prompt_step3)
    return adjustment

def step_4(adjustment, target_passage, label):
    # Step 4: Apply the adjustments to the original passage
    # label: 0 for negative stance, 1 for positive stance
    stance = "negative" if label == 0 else "positive"

    prompt_step4 = f'''You are an AI assistant tasked with applying specific adjustments to a passage.

**Original Passage:**
{target_passage}

**Adjustment Suggestions:**
{adjustment}

...
**Output Format:**
[Modified Passage]

Please return the modified passage directly without any other content.and don't output in the format of "[Modified Passage]/n...",directly output the modified passage

'''
    modified_passage = gpt(prompt_step4)
    return modified_passage

def know_attack_augment(query_list, target_passage, label=1, topk=5, num=3):
    keywords = step_1_phrase(topk, query_list)
    analysis = step_2(keywords, target_passage)
    adjustment = step_3_augment(analysis, target_passage, num, label)
    passage1 = step_4(adjustment, target_passage, label)
    return {
        "keywords": keywords,
        "analysis": analysis,
        "adjustment": adjustment,
        "passage": passage1,
    }

def know_attack_restrict(query_list, target_passage, label=1, topk=5, num=3):
    keywords = step_1_phrase(topk, query_list)
    analysis = step_2(keywords, target_passage)
    adjustment = step_3_restrict(analysis, target_passage, num, label)
    passage1 = step_4(adjustment, target_passage, label)
    return {
        "keywords": keywords,
        "analysis": analysis,
        "adjustment": adjustment,
        "passage": passage1,
    }

def sample_passage(query_list, target_passage, label=0, iter=5, topk=10, num=2, augment=True):
    passage_list, all_data = [], []
    for _ in range(iter):
        output = know_attack_augment(query_list, target_passage, label, topk, num) if augment else know_attack_restrict(query_list, target_passage, label, topk, num)
        passage_list.append(output['passage'])
        all_data.append(output)
    return passage_list, all_data

def final_attack(query_list, target_passage, label, iter=3, N=5, topk=10, num=3):
    pas_list, data_process = [], []
    for _ in range(N):
        gen_list, data_1 = sample_passage(query_list, target_passage, label, iter, topk, num)
        data_process.extend(data_1)
        filter_list = filter_passage(gen_list, target_passage)
        if filter_list:
            curr_best = find_best_passage(filter_list, target_passage, query_list)
            pas_list.extend(filter_list)
    return pas_list, data_process

def opinion_attack(passage_list, query_list, label):
    attack_passages = []
    for i, passage in enumerate(passage_list):
        if len(attack_passages) == 5:
            break
        ori_score = get_scores_nbbert(query_list, passage[1])
        pas_list, _ = final_attack(query_list, passage[1], label, 3, 5, 10, 3)
        best_pas = find_best_passage(pas_list, passage[1], query_list)
        attack_passages.append((best_pas, passage[0]))
    return [{"num": p[1], "attack_passage": p[0]} for p in attack_passages]

def attack_topic(path, topic_id, stance=0):  # stance: 0 = oppose, 1 = support
    with open(path, "r") as json_file:
        data = json.load(json_file)

    example = data[topic_id]
    query_list = example['queries']
    passage_list_0, passage_list_1 = [], []
    for item in example['passages']:
        (passage_list_0 if item[1] == 0 else passage_list_1).append((item[3], item[0]))

    passage_list_0 = sort_passages(query_list, passage_list_0)
    passage_list_1 = sort_passages(query_list, passage_list_1)

    if stance == 0 and len(passage_list_0) >= 5:
        attack_data = opinion_attack(passage_list_0, query_list, 0)
        with open(f"opinion_know_attack_data_{topic_id}_0.json", "w") as f:
            json.dump(attack_data, f, indent=4)
    elif stance == 1 and len(passage_list_1) >= 5:
        attack_data = opinion_attack(passage_list_1, query_list, 1)
        with open(f"opinion_know_attack_data_{topic_id}_1.json", "w") as f:
            json.dump(attack_data, f, indent=4)