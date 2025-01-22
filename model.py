import torch
import torch.nn as nn

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class CNN(nn.Module):
    def __init__(self, image_size, num_classes):
        super(CNN, self).__init__()
        self.filter = 64
        self.fc_hidden = 256
        self.layers = nn.Sequential(
            nn.Conv2d(1, self.filter, 3, 1, 1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter, self.filter, 3, 1, 1),
            nn.BatchNorm2d(self.filter),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.filter, self.filter*2, 3, 1, 1),
            nn.BatchNorm2d(self.filter*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.filter*2, self.filter * 2, 3, 1, 1),
            nn.BatchNorm2d(self.filter * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.output_size = image_size//4
        self.fc1 = nn.Linear(self.filter*2 * self.output_size * self.output_size, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
class CNN1 (nn.Module):
    def __init__(self, image_size, num_classes):
        super(CNN, self).__init__()
        self.filter_size = 64
        self.image_size = image_size // 4
        self.hidden_size = 256
        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.bn12 = nn.BatchNorm2d(self.filter_size)

        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size*2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.filter_size*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size * self.image_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv12(output)
        output = self.bn12(output)
        output = self.relu(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.pool(output)

        output = self.fc1(output.view(output.size(0), -1))
        output = self.fc2(output)
        return output

class CRNN (nn.Module):
    def __init__(self, input_height, num_classes):
        super(CRNN, self).__init__()
        self.filter_size = 64
        self.image_size = input_height // 4
        self.hidden_size = 256

        self.lstm_hidden = 256
        self.lstm_layer = 2

        self.attention_head = 8

        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.bn12 = nn.BatchNorm2d(self.filter_size)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 1, 1)
        self.conv22 = nn.Conv2d(self.filter_size *2, self.filter_size*2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.filter_size * 2)
        self.bn22 = nn.BatchNorm2d(self.filter_size*2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size, self.hidden_size)

        self.lstm1 = nn.LSTM(self.hidden_size, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.lstm_hidden*2, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional= True, batch_first=True)

        self.attention = nn.MultiheadAttention(self.lstm_hidden*2, self.attention_head, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(self.lstm_hidden*2)
        self.layer_norm2 = nn.LayerNorm(self.lstm_hidden*2)
        self.fc2 = nn.Linear(self.lstm_hidden*2, num_classes+1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv12(output)
        output = self.bn12(output)
        output = self.relu(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv22(output)
        output = self.bn22(output)
        output = self.relu(output)
        output = self.pool(output)

        bs, c, h, w = output.size()
        output = output.permute(0, 3, 1, 2)
        output = output.view(bs, w, -1)

        output = self.fc1(output)

        output1, _ = self.lstm1(output)
        output1 = self.layer_norm1(output1)

        attention_out, _ = self.attention(output1, output1, output1)

        output2, _ = self.lstm2(attention_out)
        output2 = self.layer_norm2(output2)

        output = output1 + output2
        output = self.fc2(output)
        return output




class CNN2(nn.Module):
    def __init__(self, image_size, classes):
        super(CNN2, self).__init__()
        self.filter_size = 64
        self.image_size = image_size // 4
        self.classes = classes
        self.hidden_size = 128

        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 2, 1)
        self.conv4 = nn.Conv2d(self.filter_size*2, self.filter_size*2, 3, 1,1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear((self.filter_size * 2) * (self.image_size*self.image_size), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.classes)
        #self.softmax = nn.Softmax()
    def forward(self, x):
        #initial residual convolution
        output1 = self.conv1(x)
        output1 = self.relu(output1)
        output2 = self.conv2(output1)
        output2 = self.relu(output2)
        output = output1 + output2
        output = self.relu(output)

        #output = self.pool(output2)
        #convolutional downsample
        output = self.conv3(output)
        output = self.relu(output)

        #normal convolution
        output = self.conv4(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.fc1(output.view(output.size(0), -1))
        output = self.fc2(output)
        return output

class convoLayer (nn.Module):
    def __init__(self):
        super(convoLayer, self).__init__()
        self.filter_size = 64
        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 2, 1)
        self.conv4 = nn.Conv2d(self.filter_size * 2, self.filter_size * 2, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #initial residual convolution
        output1 = self.conv1(x)
        output1 = self.relu(output1)
        output2 = self.conv2(output1)
        output2 = self.relu(output2)
        output = output1 + output2
        output = self.relu(output)

        #output = self.pool(output)
        #convolutional downsample
        output = self.conv3(output)
        output = self.relu(output)

        #normal convolution
        output = self.conv4(output)
        output = self.relu(output)
        output = self.pool(output)
        #print(output.size())
        #output = self.fc1()
        return output
class CNNwithTransferLearning(nn.Module):
    def __init__(self, image_size, classes, device=torch.device("cpu")):
        super(CNNwithTransferLearning, self).__init__()
        self.filter_size = 64
        self.image_size = image_size // 4
        self.classes = classes
        self.hidden_size = 128
        self.device = device
        checkpoint = torch.load("ETL2.pth")
        self.conv = convoLayer(self.image_size).to(self.device)
        self.conv.load_state_dict(checkpoint["model"], strict=False)
        self.fc1 = nn.Linear((self.filter_size * 2) * (self.image_size * self.image_size), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.classes)
    def forward(self, x):
        output = self.conv(x)
        print(output.size())
        output = self.fc1(output.view(output.size(0), -1))

        output = self.fc2(output)
        return output

class CRNN2(nn.Module):
    def __init__(self):
        super(CRNN2, self).__init__()
        pass


class CRNN1(nn.Module):
    def __init__(self, input_height=32, num_classes=2168):
        super(CRNN1, self).__init__()
        self.cnn = convoLayer()
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),


        )
        """



        # Calculate input size for LSTM based on CNN output
        lstm_input_size = 128 * (input_height // 4)  # Assume input height = 32, //4 due to pooling
        self.hidden_size = 256

        #self.map_to_seq = nn.Linear(lstm_input_size, self.hidden_size)

        # Attention layer (Multihead)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size*2,
            num_heads=8,
            batch_first=False  # Keep it False, transpose manually before attention
        )

        # Bidirectional LSTMs
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,


        )


        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size*2,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,


        )

        self.layer_norm1 = nn.LayerNorm(self.hidden_size*2)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size*2)
        #self.dropout = nn.Dropout(0.5)

        #self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size*2, num_classes + 1)
        #self.log_softmax = nn.LogSoftmax(2)


    def _init_hidden(self, batch_size, device):
        """Initialize hidden states deterministically during evaluation"""
        h0 = torch.zeros(4, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(4, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, x, deterministic=False):
        device = x.device

        if deterministic:
            # Set seeds for reproducibility
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(0)
                torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # CNN forward pass


        output = self.cnn(x)



        bs, c, h, w = output.size()

        # Reshape for LSTM

        output = output.permute(0, 3, 1, 2)

        output = output.view(bs, w, -1)

        #Feature Map to Sequence
        #output = self.map_to_seq(output)

        # Initialize hidden states
        hidden1 = self._init_hidden(bs, device)
        hidden2 = self._init_hidden(bs, device)

        # First LSTM
        lstm1_out, _ = self.lstm(output, hidden1)
        lstm1_out = self.layer_norm1(lstm1_out)

        # Self-attention
        lstm1_out_transpose = lstm1_out.transpose(0, 1)  # Transpose for attention (seq_len, batch_size, embed_dim)
        attended_out, _ = self.attention(
            lstm1_out_transpose,
            lstm1_out_transpose,
            lstm1_out_transpose
        )
        attended_out = attended_out.transpose(0, 1)  # Transpose back (batch_size, seq_len, embed_dim)

        # Second LSTM
        lstm2_out, _ = self.lstm2(attended_out, hidden2)
        lstm2_out = self.layer_norm2(lstm2_out)

        # Combine LSTM outputs
        output = lstm1_out + lstm2_out

        #if self.training and not deterministic:
        #    output = self.dropout(output)

        output = self.fc1(output)
        #output = self.log_softmax(output)

        return output
