# %%
import numpy as np
import os
# import h5py
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from scipy import fft
import torch.fft
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
device = torch.device("cuda:{}" .format(0) if torch.cuda.is_available() else "cpu")

log_file_path = "logs.txt"




def generate_conventional_pulse(center_freq, bandwidth, pulse_width, signal_length):
    # Generate a conventional pulse signal
    t = np.arange(signal_length) # time vector
    signal = np.zeros((signal_length)) # initialize signal array
    signal[t < pulse_width] = np.sin(2*np.pi*center_freq*t[t < pulse_width])
    signal[t >= pulse_width] = np.sin(2*np.pi*(center_freq + bandwidth)*t[t >= pulse_width])
    return signal

def generate_linear_fm(center_freq, bandwidth, pulse_width, mod_slope, signal_length):
    # Generate a linear frequency modulation signal
    t = np.arange(signal_length) # time vector
    signal = np.zeros((signal_length)) # initialize signal array
    signal[t < pulse_width] = np.sin(2*np.pi*(center_freq + bandwidth*t[t < pulse_width]/pulse_width)*t[t < pulse_width])
    signal[t >= pulse_width] = np.sin(2*np.pi*(center_freq + bandwidth + mod_slope*(t[t >= pulse_width] - pulse_width))*t[t >= pulse_width])
    return signal

def generate_phase_mod(center_freq, bandwidth, pulse_width, mod_depth, signal_length):
    # Generate a phase modulation signal
    t = np.arange(signal_length) # time vector
    signal = np.zeros((signal_length)) # initialize signal array
    signal[t < pulse_width] = np.sin(2*np.pi*(center_freq + bandwidth*t[t < pulse_width]/pulse_width)*t[t < pulse_width])
    signal[t >= pulse_width] = np.sin(2*np.pi*(center_freq + bandwidth + mod_depth*np.sin(2*np.pi*mod_depth*(t[t >= pulse_width] - pulse_width)/pulse_width))*t[t >= pulse_width])
    return signal

# Define the generate_signal function
def generate_signal(label):
    # Extract the label information
    center_freq = label['center_freq']
    bandwidth = label['bandwidth']
    mod_slope = label['mod_slope']
    code_length = label['code_length']
    interference_freq = label['interference_freq']
    fm_bandwidth = label['fm_bandwidth']
    comb_num = label['comb_num']
    comb_spacing = label['comb_spacing']
    
    modulation_type = label['modulation_type']
    interference_type = label['interference_type']
    
    # Generate a time domain signal based on the label information
    signal_length = 1000 # length of each generated signal
        # create the time-domain signal
    t = np.linspace(0, 2.5, signal_length) #signal_length/fs fs=400MHz
    signal = np.sin(2*np.pi*center_freq*t)
    
    # add echo signal if present
    if modulation_type != 0:
        if modulation_type == 1:
            echo_signal = np.sin(2*np.pi*(center_freq+bandwidth)*t)
        elif modulation_type == 2:
            mod_signal = np.linspace(0, mod_slope*2.5, signal_length)
            echo_signal = np.sin(2*np.pi*(center_freq+mod_signal)*t)
        elif modulation_type == 3:
            code = np.random.randint(0, 2, code_length)
            code_signal = np.repeat(code, signal_length//code_length+1)[:signal_length]
            echo_signal = signal * code_signal
        signal += echo_signal
    
    # add interference signal if present
    if interference_type != 0:
        if interference_type == 1:
            interference_signal = np.sin(2*np.pi*interference_freq*t)
            interference_signal *= np.random.uniform(0.1, 1)
        elif interference_type == 2:
            mod_signal = np.linspace(0, fm_bandwidth*2.5, signal_length)
            interference_signal = np.sin(2*np.pi*(interference_freq+mod_signal)*t)
        signal += interference_signal
    
    # add comb spectrum if present
    if comb_num != 0:
        comb_freqs = np.linspace(center_freq-((comb_num-1)*comb_spacing/2), center_freq+((comb_num-1)*comb_spacing/2), comb_num)
        comb_signal = np.zeros(signal_length)
        for freq in comb_freqs:
            comb_signal += np.sin(2*np.pi*freq*t)
        signal += comb_signal
    
    # add noise
    signal += np.random.normal(0, 0.1, signal_length)

    return signal

def generate_data(num_samples):
    # Initialize empty lists to store data and labels
    data = []
    labels = []

    # define the range of each label
    
    for i in range(num_samples):
        # Generate random values for all parameters within specified ranges
        center_freq = np.random.uniform(10, 200) # center frequency in MHz
        bandwidth = np.random.uniform(1, 10) # bandwidth is a fraction of center frequency in MHz    
        mod_slope = np.random.uniform(10, 1000) # slope of linear frequency modulation in MHz/us
        code_length = np.random.randint(0,4) # length of code in bits

        interference_freq = np.random.uniform(10, 200) # interference frequency in MHz
        fm_bandwidth = np.random.uniform(1, 10) # bandwidth of frequency modulation in MHz
        comb_num = np.random.randint(2, 11) # number of frequency components in frequency comb
        comb_spacing = np.random.uniform(1, 10) # spacing between frequency components in frequency comb in MHz

        modulation_type = np.random.randint(4)
        interference_type = np.random.randint(3)
        

        if modulation_type != 1:
            bandwidth = 0

        if modulation_type != 2:
            mod_slope = 0

        if modulation_type != 3:
            code_length = 0

        if modulation_type == 3:
            code_length = 2**code_length

        if interference_type == 0:
            interference_freq = 0

        if interference_type != 2:
            fm_bandwidth = 0

        if comb_num == 0:
            comb_spacing = 0

        
        # Create a dictionary to store the label information
        label = {
            'center_freq': center_freq,
            'bandwidth': bandwidth,
            'mod_slope': mod_slope,
            'code_length': code_length,
            'interference_freq': interference_freq,
            'fm_bandwidth': fm_bandwidth,
            'comb_num': comb_num,
            'comb_spacing': comb_spacing,
            'modulation_type': modulation_type,
            'interference_type': interference_type
        }
        
        
        # Append the label to the list of labels
        labels.append(label)
        
        # Generate a time domain signal based on the label information
        # (code for this is not provided)
        signal = generate_signal(label)
        
        # Append the signal to the list of data
        data.append(signal)
    
    return data, labels

    

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # time domain feature extraction
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2)
        
        # frequency domain feature extraction
        self.fft = torch.fft.fft
        self.conv4 = torch.nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride=2)
        self.relu4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv5 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.relu5 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv6 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu6 = torch.nn.ReLU()
        self.pool6 = torch.nn.MaxPool1d(kernel_size=2)
        
        # full connection
        self.fc1 = torch.nn.Linear(in_features=1920, out_features=2048)
        self.relu_fc = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=1024)

        # regression
        self.reg1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.relu_reg1 = torch.nn.ReLU()
        self.reg2 = torch.nn.Linear(in_features=512, out_features=256)
        self.relu_reg2 = torch.nn.ReLU()
        self.reg3 = torch.nn.Linear(in_features=256, out_features=8)


        # classification1
        self.class1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.relu_cls1 = torch.nn.ReLU()
        self.class2 = torch.nn.Linear(in_features=512, out_features=256)
        self.relu_cls2 = torch.nn.ReLU()
        self.class3 = torch.nn.Linear(in_features=256, out_features=4)

        # classification2
        self.class4 = torch.nn.Linear(in_features=1024, out_features=512)
        self.relu_cls4 = torch.nn.ReLU()
        self.class5 = torch.nn.Linear(in_features=512, out_features=256)
        self.relu_cls5 = torch.nn.ReLU()
        self.class6 = torch.nn.Linear(in_features=256, out_features=3)


        
    def forward(self, x):
        x = x.unsqueeze(1)

        # time domain feature extraction
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.pool2(x1)
        x1 = self.conv3(x1)
        x1 = self.relu3(x1)
        x1 = self.pool3(x1)
        
        # frequency domain feature extraction
        x2 = self.fft(x)
        x2 = torch.cat([x2.real, x2.imag], dim=1)
        x2 = self.conv4(x2)
        x2 = self.relu4(x2)
        x2 = self.pool4(x2)
        x2 = self.conv5(x2)
        x2 = self.relu5(x2)
        x2 = self.pool5(x2)
        x2 = self.conv6(x2)
        x2 = self.relu6(x2)
        x2 = self.pool6(x2)
        
        # concat
        x = torch.cat((x1, x2), dim=1)
        
        # full connection
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x= self.fc2(x)

        # regression
        reg = self.reg1(x)
        reg = self.relu_reg1(reg)
        reg = self.reg2(reg)
        reg = self.relu_reg2(reg)
        reg = self.reg3(reg)

        # classification1
        cls1 = self.class1(x)
        cls1 = self.relu_cls1(cls1)
        cls1 = self.class2(cls1)
        cls1 = self.relu_cls2(cls1)
        cls1 = self.class3(cls1)

        # classification2
        cls2 = self.class4(x)
        cls2 = self.relu_cls4(cls2)
        cls2 = self.class5(cls2)
        cls2 = self.relu_cls5(cls2)
        cls2 = self.class6(cls2)
        
        return reg, cls1, cls2 
    
def train(net, train_loader, val_loader, epochs, optimizer, cls_criterion, reg_criterion, train_labels_mean, train_labels_std):
    train_reg_loss, train_cls1_loss, train_cls2_loss = [], [], []
    val_reg_loss, val_cls1_loss, val_cls2_loss = [], [], []
    train_acc1, train_acc2 = [], []
    val_acc1, val_acc2 = [], []

    for epoch in range(epochs):
        net.train()
        running_reg_loss = 0.0
        running_cls1_loss = 0.0
        running_cls2_loss = 0.0
        running_acc1= 0.0
        running_acc2= 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            reg, cls1, cls2 = labels[:, :-2], labels[:, -2], labels[:, -1]
            inputs = inputs.float().to(device)
            reg = reg.float().to(device)
            cls1 = cls1.long().to(device)
            cls2 = cls2.long().to(device)
            optimizer.zero_grad()
            outputs, cls_out1, cls_out2 = net(inputs)
            outputs = outputs*train_labels_std + train_labels_mean
            reg_loss = reg_criterion(outputs, reg)
            cls1_loss = cls_criterion(cls_out1, cls1)
            cls2_loss = cls_criterion(cls_out2, cls2)
            (reg_loss+cls1_loss+cls2_loss).backward()
            optimizer.step()
            
            running_reg_loss += reg_loss.item()
            running_cls1_loss += cls1_loss.item()
            running_cls2_loss += cls2_loss.item()
            _, preds1 = torch.max(cls_out1, 1)
            _, preds2 = torch.max(cls_out2, 1)
            running_acc1 += accuracy_score(preds1.cpu().numpy(), cls1.cpu().numpy())
            running_acc2 += accuracy_score(preds2.cpu().numpy(), cls2.cpu().numpy())

        train_reg_loss.append(running_reg_loss/len(train_loader))
        train_cls1_loss.append(running_cls1_loss/len(train_loader))
        train_cls2_loss.append(running_cls2_loss/len(train_loader))
        train_acc1.append(running_acc1/len(train_loader))
        train_acc2.append(running_acc2/len(train_loader))
        

        net.eval()
        val_running_reg_loss = 0.0
        val_running_cls1_loss = 0.0
        val_running_cls2_loss = 0.0
        val_running_acc1 = 0.0
        val_running_acc2 = 0.0

        for i, data in enumerate(val_loader):
            inputs, labels = data
            reg, cls1, cls2 = labels[:, :-2], labels[:, -2], labels[:, -1]
            inputs = inputs.float().to(device)
            reg = reg.float().to(device)
            cls1 = cls1.long().to(device)
            cls2 = cls2.long().to(device)
            optimizer.zero_grad()
            outputs, cls_out1, cls_out2 = net(inputs)
            outputs = outputs*train_labels_std + train_labels_mean
            reg_loss = reg_criterion(outputs, reg)
            cls1_loss = cls_criterion(cls_out1, cls1)
            cls2_loss = cls_criterion(cls_out2, cls2)

            val_running_reg_loss += reg_loss.item()
            val_running_cls1_loss += cls1_loss.item()
            val_running_cls2_loss += cls2_loss.item()
            _, preds1 = torch.max(cls_out1, 1)
            _, preds2 = torch.max(cls_out2, 1)
            val_running_acc1 += accuracy_score(preds1.cpu().numpy(), cls1.cpu().numpy())
            val_running_acc2 += accuracy_score(preds2.cpu().numpy(), cls2.cpu().numpy())

        val_reg_loss.append(val_running_reg_loss/len(val_loader))
        val_cls1_loss.append(val_running_cls1_loss/len(val_loader))
        val_cls2_loss.append(val_running_cls2_loss/len(val_loader))
        val_acc1.append(val_running_acc1/len(val_loader))
        val_acc2.append(val_running_acc2/len(val_loader))

        print('Epoch [{}/{}], train_reg_loss: {:.4f},  train_cls1_loss: {:.4f}, train_cls2_loss: {:.4f}, train_acc1: {:.4f}, train_acc2: {:.4f}'.format(epoch+1, epochs, train_reg_loss[-1], train_cls1_loss[-1], train_cls2_loss[-1], train_acc1[-1], train_acc2[-1]))
        print('Epoch [{}/{}], val_reg_loss: {:.4f},  val_cls1_loss: {:.4f}, val_cls2_loss: {:.4f}, val_acc1: {:.4f}, val_acc2: {:.4f}'.format(epoch+1, epochs, val_reg_loss[-1], val_cls1_loss[-1], val_cls2_loss[-1], val_acc1[-1], val_acc2[-1]))
        with open(log_file_path,'a+') as f:
            f.write('Epoch [{}/{}], train_reg_loss: {:.4f},  train_cls1_loss: {:.4f}, train_cls2_loss: {:.4f}, train_acc1: {:.4f}, train_acc2: {:.4f}\n'.format(epoch+1, epochs, train_reg_loss[-1], train_cls1_loss[-1], train_cls2_loss[-1], train_acc1[-1], train_acc2[-1]))
            f.write('Epoch [{}/{}], val_reg_loss: {:.4f},  val_cls1_loss: {:.4f}, val_cls2_loss: {:.4f}, val_acc1: {:.4f}, val_acc2: {:.4f}\n'.format(epoch+1, epochs, val_reg_loss[-1], val_cls1_loss[-1], val_cls2_loss[-1], val_acc1[-1], val_acc2[-1]))

    return train_reg_loss, train_cls1_loss, train_cls2_loss, train_acc1, train_acc2, val_reg_loss, val_cls1_loss, val_cls2_loss, val_acc1, val_acc2


def test(net, test_loader, cls_criterion, reg_criterion, train_labels_mean, train_labels_std):
    net.eval()
    test_running_reg_loss = 0.0
    test_running_cls1_loss = 0.0
    test_running_cls2_loss = 0.0
    test_running_acc1 = 0.0
    test_running_acc2 = 0.0

    for i, data in enumerate(test_loader):
        inputs, labels = data
        reg, cls1, cls2 = labels[:, :-2], labels[:, -2], labels[:, -1]
        inputs = inputs.float().to(device)
        reg = reg.float().to(device)
        cls1 = cls1.long().to(device)
        cls2 = cls2.long().to(device)
        optimizer.zero_grad()
        outputs, cls_out1, cls_out2 = net(inputs)
        outputs = outputs*train_labels_std + train_labels_mean
        reg_loss = reg_criterion(outputs, reg)
        cls1_loss = cls_criterion(cls_out1, cls1)
        cls2_loss = cls_criterion(cls_out2, cls2)

        test_running_reg_loss += reg_loss.item()
        test_running_cls1_loss += cls1_loss.item()
        test_running_cls2_loss += cls2_loss.item()
        _, preds1 = torch.max(cls_out1, 1)
        _, preds2 = torch.max(cls_out2, 1)
        test_running_acc1 += accuracy_score(preds1.cpu().numpy(), cls1.cpu().numpy())
        test_running_acc2 += accuracy_score(preds2.cpu().numpy(), cls2.cpu().numpy())

    test_reg_loss = test_running_reg_loss/len(test_loader)
    test_cls1_loss = test_running_cls1_loss/len(test_loader)
    test_cls2_loss = test_running_cls2_loss/len(test_loader)
    test_acc1 = test_running_acc1/len(test_loader)
    test_acc2 = test_running_acc2/len(test_loader)

    print('Test reg loss: {:.4f}, Test cls1 loss: {:.4f}, Test cls2 loss: {:.4f}, Test acc1: {:.4f}, Test acc2: {:.4f}'.format(test_reg_loss, test_cls1_loss, test_cls2_loss, test_acc1, test_acc2))
    with open(log_file_path,'a+') as f:
        f.write('Test reg loss: {:.4f}, Test cls1 loss: {:.4f}, Test cls2 loss: {:.4f}, Test acc1: {:.4f}, Test acc2: {:.4f}\n'.format(test_reg_loss, test_cls1_loss, test_cls2_loss, test_acc1, test_acc2))

    return test_reg_loss, test_cls1_loss, test_cls2_loss, test_acc1, test_acc2

    

def load_test(net, cls_criterion, reg_criterion, train_data, train_labels):
    net.load_state_dict(torch.load('multi_model_large.pth', map_location=device))
    print("loaded model")

    train_labels_mean = torch.tensor(np.mean(train_labels[:,:-2], axis=0)).float().to(device)
    train_labels_std = torch.tensor(np.std(train_labels[:,:-2], axis=0)).float().to(device)

    # Test set data loader
    test_set = data_utils.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Test model
    test(net, test_loader, cls_criterion, reg_criterion, train_labels_mean, train_labels_std)


if __name__ == '__main__':
    data_num = 1000000

    # Check if data is already generated
    if os.path.exists('data_large.pkl') and os.path.exists('labels_large.pkl'):
        # with open('data_large.pkl', 'rb') as f:
        #     data = pickle.load(f)
        # with open('labels_large.pkl', 'rb') as f:
        #     labels = pickle.load(f)

        data, labels = generate_data(data_num)
        print('{} Data generated'.format(data_num))
    else:
        # Generate 1000000 samples of data
        data, labels = generate_data(data_num)
        print('Data generated')

        # Save the data and labels to a file
        with open('data_large.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open('labels_large.pkl', 'wb') as f:
            pickle.dump(labels, f)

    # Define hyperparameters
    epochs = 50
    batch_size = 128
    learning_rate = 1e-4

    # define loss function and optimizer
    net = Net().to(device)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



    # Divide the dataset into training, validation, and testing sets.
    data = np.array(data)
    labels = np.array([list(d.values()) for d in labels])

    if os.path.exists('multi_model_large.pth'):
        load_test(net, cls_criterion, reg_criterion, data, labels)
        exit()

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)
    train_labels_mean = torch.tensor(np.mean(train_labels[:,:-2], axis=0)).float().to(device)
    train_labels_std = torch.tensor(np.std(train_labels[:,:-2], axis=0)).float().to(device)

    # Training set data loader
    train_set = data_utils.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Validation set data loader
    val_set = data_utils.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels))
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Test set data loader
    test_set = data_utils.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Train model
    train(net, train_loader, val_loader, epochs, optimizer, cls_criterion, reg_criterion, train_labels_mean, train_labels_std)

    # Test model
    test(net, test_loader, cls_criterion, reg_criterion, train_labels_mean, train_labels_std)

    # Save model
    torch.save(net.state_dict(), 'multi_model_large.pth')
    print('Model saved')    


