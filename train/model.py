import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
        self.word_dict = None
        self.hidden_dim = hidden_dim
        
    def forward(self, inputs):
        """
        Perform a forward pass of our model on some input.
        """
        
        # transpose the inputs
        inputs = inputs.t()
        
        # separate the reviews and the review lengths
        review_lengths = inputs[0,:]
        reviews = inputs[1:,:]
        
        # ensure embedding layer gets a LongTensor input
        reviews = reviews.long()
        embeds = self.embedding(reviews)
        
        # LSTM
        output, _ = self.lstm(embeds)      
        
        # pass through the full connected layer
        output = self.dense(output)
        output = output[review_lengths - 1, range(len(review_lengths))]
        
        # pass the raw logits through sigmoid activation layer
        output = self.sigmoid(output)
        
        return output.squeeze()