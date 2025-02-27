import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # Initialize weights for update gate, reset gate, and candidate hidden state
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Initialize biases
        self.update_gate.bias.data.fill_(0)
        self.reset_gate.bias.data.fill_(0)
        self.candidate.bias.data.fill_(0)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Calculate update and reset gates
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        # Calculate candidate hidden state
        reset_hidden = reset * h
        candidate_input = torch.cat([x, reset_hidden], dim=1)
        candidate_hidden = torch.tanh(self.candidate(candidate_input))
        
        # Calculate new hidden state
        new_h = (1 - update) * h + update * candidate_hidden
        
        return new_h


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs shape: (batch_size, sequence_length, input_size)
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        # Process sequence
        for t in range(seq_length):
            h = self.cell(inputs[:, t, :], h)
            
        # Return final hidden state
        return h