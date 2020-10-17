import torch.nn as nn


class NER_LSTM(nn.Module):
    def __init__(   self, input_size, embedding_size, hidden_size, output_size,
                    num_layers, num_dirs,
                    dropout_,
                ):

        super().__init__()

        self.dropout = nn.Dropout(dropout_)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM( embedding_size, 
                            hidden_size, 
                            num_layers=num_layers,
                            bidirectional=num_dirs,
                            dropout= dropout_ if num_layers > 1 else 0,
                            batch_first=True
                        )
        self.fc = nn.Linear(
            hidden_size * 2 if num_dirs else hidden_size, output_size)

    def forward(self, x):
        
        # x shape: (batch_size, seq_length)
        # embedding shape: (batch_size, seq_length, embedding_size)
        embedding = self.dropout(self.embedding(x))

        # outputs shape: (num_batch, seq_length, hidden_size*2)
        outputs, _ = self.rnn(embedding)

        # outputs shape: (num_batch, seq_length, output_size)
        predictions = self.fc(self.dropout(outputs))
        
        return predictions
