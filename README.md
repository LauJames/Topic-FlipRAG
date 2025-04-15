# Topic-FlipRAG:Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models


![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of Topic-FlipedRAG, a novel two-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems. The proposed method demonstrates how strategic knowledge poisoning can systematically manipulate LLM outputs for opinion-oriented tasks through semantic-level perturbations.

## Key Features
- 🎯 **Topic-oriented attacks** on multi-perspective content generation
- ⚡ **Dual-phase manipulation** combining:
  - Traditional adversarial ranking techniques
  - LLM-driven semantic perturbation generation
- 📊 Comprehensive evaluation framework for opinion shift measurement

## Installation

```bash
git clone https://github.com/your_anonymous_repo/Topic-FlipedRAG.git
cd Topic-FlipedRAG
pip install -r requirements.txt
```

**Requirements**:
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- FAISS 1.7.2+
- (Complete with your actual dependencies)

## Datasets
### MsMarco
**overview:**
The MAchine Reading COmprehension ([MSMARCO](https://microsoft.github.io/msmarco/)) dataset  is based on sampled real users' Bing queries. The corpus is initially constructed by retrieving the top-10 passages from the Bing search engine and then annotated. Relevance labels are sparsely-judged and derived from what passages are marked as having the answer to the query. The full training set contains approximately 400M tuples of a query, relevant and non-relevant passages. The development set (MSMARCO DEV) of passage reranking contains 6,980 queries, each paired with the top 1,000 passages retrieved with BM25 from the MSMARCO corpus.

**topic-queries generation procession:**
To construct topic-lists for evaluation, we applied a Kmeans clustering algorithm to group similar queries, forming topics that each contained a series of related queries. To further evaluate the performance of our method under extreme topic-query scenarios, we applied an intra-topic similarity filtering process. Only topics with queries exhibiting high semantic diversity and containing a sufficient number of queries were retained.
This process resulted in 29 topics, with each topic containing an average of 22.28 queries. The average similarity score within each topic was approximately 0.5, indicating sufficient diversity among queries to ensure a rigorous evaluation

### PROCON 
**overview:**
To conduct our experiments, we utilized controversial topic data scraped from the PROCON.ORG website.The controversial topic dataset includes over 80 topics,covering fields such as society, health, government, and education.Each controversial topic is discussed from two stances (Pro and Con), with an average of 30 related passages, each holding a certain opinion with stance Pro or Con.

**topic-queries generation procession:**
To simulate real-world user interactions with a RAG system, we instructed a large language model (GPT-4o) to act as a user and generate 40 potential sub-queries for each topic.These sub-queries were designed to reflect the diverse questions and concerns users might raise when exploring a specific controversial topic.After generating the sub-queries, we applied a similarity filtering process to ensure diversity by retaining only those with a similarity score below approximately 0.85. The filtering step effectively removed redundant queries while preserving
a wide range of perspectives. As a result, the final set of topicqueries achieved an average similarity score of approximately 0.7, ensuring that the queries were sufficiently diverse yet semantically relevant. 



## Usage

### Basic Attack Pipeline
```python
from attack_pipeline import TopicFlipedRAG

# Initialize attack module
attack_config = {
    "target_topic": "climate_change",
    "opinion_direction": "skepticism",
    "perturbation_level": 0.3
}
attacker = TopicFlipedRAG(**attack_config)

# Execute attack on RAG system
compromised_responses = attacker.execute_attack(
    base_retriever=your_retriever,
    generator_model=your_llm,
    query_batch=test_queries
)
```

### Evaluation Metrics
```python
from evaluation import OpinionShiftAnalyzer

analyzer = OpinionShiftAnalyzer(reference_corpus="neutral_responses.json")
shift_scores = analyzer.calculate_opinion_shift(
    original_responses=baseline_outputs,
    attacked_responses=compromised_responses
)
```

## Experimental Results

Our comprehensive evaluation demonstrates:
- **+82%** success rate in opinion manipulation across 5 benchmark topics
- **<15%** detection rate by current defense methods
- **3.2x** amplification effect in multi-query scenarios

(Replace with your actual experimental metrics)

## Contributing

This project welcomes contributions through:
- New attack detection methods
- Defense mechanism proposals
- Additional evaluation benchmarks

Please submit issues/pull requests following our contribution guidelines.

## License

MIT License (see [LICENSE](LICENSE) for details)

---

**Disclaimer**: This implementation is provided for research purposes only. Users must adhere to ethical AI guidelines and applicable laws when using this code.
