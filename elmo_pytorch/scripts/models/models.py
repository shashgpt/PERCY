from config import *



class cnn(nn.Module):
    def __init__(self, config):
        super(cnn, self).__init__()
        self.config = config
        self.elmo = self.config["elmo_model"]
        for param in self.elmo.parameters():
            param.requires_grad = self.config["fine_tune_word_embeddings"]
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = self.config["n_filters"], kernel_size = (self.config["filter_sizes"][0], self.config["embedding_dim"]))
        self.conv_2 = nn.Conv2d(in_channels = 1, out_channels = self.config["n_filters"], kernel_size = (self.config["filter_sizes"][1], self.config["embedding_dim"]))
        self.conv_3 = nn.Conv2d(in_channels = 1, out_channels = self.config["n_filters"], kernel_size = (self.config["filter_sizes"][2], self.config["embedding_dim"]))
        self.fc = nn.Linear(len(self.config["filter_sizes"]) * self.config["n_filters"], self.config["classes"]) # Output Layer
        self.dropout = nn.Dropout(self.config["dropout"])

    def forward(self, x): #x = batch_size x sent_len x 50
        x = x.type(torch.long)
        # x_clone = x.clone().type(torch.long)
        embeddings = self.elmo(x) # x= batch_size x sent_len x 1024
        emb = embeddings['elmo_representations'][0][:,:,:self.config["embedding_dim"]]
        x = emb.unsqueeze(1) #x = batch_size x 1 x sent_len x embedding_dim
        x_1 = F.relu(self.conv_1(x).squeeze(3)) #x_n = batch size x n_filters x sent len - filter_sizes[n] + 1
        x_2 = F.relu(self.conv_2(x).squeeze(3))
        x_3 = F.relu(self.conv_3(x).squeeze(3))
        x_1 = F.max_pool1d(x_1, x_1.shape[2]).squeeze(2) #x_n = batch size x n_filters
        x_2 = F.max_pool1d(x_2, x_2.shape[2]).squeeze(2)
        x_3 = F.max_pool1d(x_3, x_3.shape[2]).squeeze(2)
        x = self.dropout(torch.cat((x_1, x_2, x_3), dim = 1)) #x = batch size x n_filters * len(filter_sizes)
        # x = torch.cat((x_1, x_2, x_3), dim = 1)
        return self.fc(x) # returns a batch_sizex2 logits vector

class lstm(nn.Module):
    def __init__(self, config):
        super(lstm, self).__init__()
        self.config = config
        self.elmo = self.config["elmo_model"]
        for param in self.elmo.parameters():
            param.requires_grad = self.config["fine_tune_word_embeddings"]
        self.lstm = nn.LSTM(input_size = self.config["embedding_dim"], hidden_size = self.config["hidden_units_seq_layer"], batch_first=True)
        self.fc = nn.Linear(self.config["hidden_units_seq_layer"], self.config["classes"]) # Output Layer

    def forward(self, x): #x = batch_size x sent_len x 50
        x = x.type(torch.long)
        # x_clone = x.clone().type(torch.long)
        embeddings = self.elmo(x) 
        emb = embeddings['elmo_representations'][0][:,:,:self.config["embedding_dim"]] # x = batch_size x sent_len x 1024
        x = self.lstm(emb)[0][:, -1, :]
        return self.fc(x) # returns a batch_sizex2 logits vector
    
    
    