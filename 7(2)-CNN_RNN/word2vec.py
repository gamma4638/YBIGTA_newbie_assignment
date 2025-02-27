import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Literal
from tqdm import tqdm


class Word2VecDataset(Dataset):
    def __init__(
        self,
        input_ids: list[list[int]],
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        self.input_ids = input_ids
        self.window_size = window_size
        self.method = method
        self.items: list[tuple[list[int], int]] = []
        
        # Create training pairs based on method
        for sentence in input_ids:
            if len(sentence) < 2 * window_size + 1:
                continue
                
            for i in range(window_size, len(sentence) - window_size):
                context = sentence[i-window_size:i] + sentence[i+1:i+window_size+1]
                target = sentence[i]
                
                if method == "cbow":
                    self.items.append((context, target))
                else:  # skipgram
                    for ctx_word in context:
                        self.items.append(([target], ctx_word))
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        context, target = self.items[index]
        return torch.tensor(context), torch.tensor(target)


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        
        # Initialize weights
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.weight.weight.data.uniform_(-0.1, 0.1)

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        
        # Tokenize corpus
        input_ids = [tokenizer(text, add_special_tokens=False)["input_ids"] 
                    for text in corpus]
        
        # Create dataset and dataloader
        dataset = Word2VecDataset(input_ids, self.window_size, self.method)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.train()
        for epoch in range(num_epochs):
            loss_sum: float = 0.0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for context, target in progress_bar:
                optimizer.zero_grad()
                
                if self.method == "cbow":
                    loss = self._train_cbow(context, target)
                else:
                    loss = self._train_skipgram(context, target)
                    
                loss.backward()
                optimizer.step()
                
                loss_sum += loss.item()
                progress_bar.set_postfix({'loss': loss_sum / (progress_bar.n + 1)})

    def _train_cbow(
        self,
        context: Tensor,  # Shape: (batch_size, 2*window_size)
        target: Tensor    # Shape: (batch_size,)
    ) -> Tensor:
        # Get context embeddings and average them
        context_embeds = self.embeddings(context)  # (batch_size, 2*window_size, d_model)
        context_mean = context_embeds.mean(dim=1)  # (batch_size, d_model)
        
        # Predict target word
        output = self.weight(context_mean)  # (batch_size, vocab_size)
        
        return nn.functional.cross_entropy(output, target)

    def _train_skipgram(
        self,
        target: Tensor,   # Shape: (batch_size, 1)
        context: Tensor   # Shape: (batch_size,)
    ) -> Tensor:
        # Get target word embedding
        target_embed = self.embeddings(target).squeeze(1)  # (batch_size, d_model)
        
        # Predict context word
        output = self.weight(target_embed)  # (batch_size, vocab_size)
        
        return nn.functional.cross_entropy(output, context)