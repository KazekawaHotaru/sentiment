import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        # Embed the input text
        embedded = self.embedding(text)

        # Pack the padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), enforce_sorted=False,
                                                            batch_first=True)

        # Pass the packed sequence through the LSTM
        output, (hidden, _) = self.rnn(packed_embedded)

        # Get the last hidden state of the LSTM
        last_hidden = hidden[-1, :, :]

        # Pass the last hidden state through the fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


# Example usage
vocab_size = 10000  # Size of the vocabulary
embed_dim = 128  # Dimension of the embeddings
rnn_hidden_size = 256  # Number of hidden units in the LSTM
fc_hidden_size = 64  # Number of hidden units in the fully connected layer

model = SentimentClassifier(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)

# Train the model on a dataset of labeled text
# ...

# Use the model to classify new text
text = torch.tensor(["This movie is great!", "This movie is terrible."])
lengths = torch.tensor([5, 7])

predictions = model(text, lengths)
print(predictions)
