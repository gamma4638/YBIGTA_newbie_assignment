
from datasets import load_dataset
from transformers import AutoTokenizer
# 구현하세요

def load_corpus() -> list[str]:
    # Load dataset from HuggingFace
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    
    # Initialize corpus list
    corpus: list[str] = []
    
    # Add texts from all splits (train, validation, test)
    for split in dataset.keys():
        texts = dataset[split]["verse_text"]
        corpus.extend(texts)
    
    return corpus