import torch.nn as nn

class LSTMClassifier(nn.Module):
    '''
    Simple single layered RNN (LSTM) model for Sentiment Analysis
    '''

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        '''
        Initialize the model by setting up the various layers
        '''
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
        self.word_dict = None
        self.hidden_dim = hidden_dim
        
    def forward(self, inputs):
        '''
        Define LSTMClassifier's forward pass
        '''
        
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
        output = output.contiguous().view(-1, self.hidden_dim)
        
        # pass through the full connected layer
        output = self.dense(output)
        
        # pass the raw logits through sigmoid activation layer
        output = self.sigmoid(output)
        
        return output.squeeze()