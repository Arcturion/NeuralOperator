import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x_data, look_back):
        self.x_data = x_data
        self.look_back = look_back

    def __len__(self):
        return len(self.x_data) - self.look_back + 1

    def __getitem__(self, idx):
        x_seq = self.x_data[idx:idx + self.look_back]
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        return x_seq.squeeze(-1).permute(0, 3, 1, 2).contiguous()

# Load your dataset
x_train = np.load('inputs.npy')[:5000]  # Shape: (19000, 64, 64, 1, 1)

train_dataset = MyDataset(x_train, 7)

batch_size = 128  # You can adjust the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




#
#MODEL
#



import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize, _, _, height, width = x.shape
        
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            torch.einsum("bijxy,ioxy->bojxy", x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bijxy,ioxy->bojxy", x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(height, width))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2

        self.fc0 = nn.Linear(6, self.width)  # Changed to 7 to account for 7 time steps
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv3d(self.width, self.width, kernel_size=(1, 1, 1))
        self.w1 = nn.Conv3d(self.width, self.width, kernel_size=(1, 1, 1))
        self.w2 = nn.Conv3d(self.width, self.width, kernel_size=(1, 1, 1))
        self.w3 = nn.Conv3d(self.width, self.width, kernel_size=(1, 1, 1))
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Input shape: (batch_size, 7, 64, 64)
        batch_size, time_steps, channels, height, width = x.shape
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.bn0(x)
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.bn1(x)
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.bn2(x)
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x)
        
        #x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.squeeze(-1)  # (batch_size, 64, 64)

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch.to(device)
            outputs = model(inputs[:, :6])
            loss = criterion(outputs, inputs[:, -1])  # Use last time step as target
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Plot the output
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(5, 4))
        #plt.imshow(outputs[0, 0, :, :].detach().cpu().numpy(), cmap='viridis')
        #plt.colorbar()
        #plt.title(f"Output at Epoch {epoch+1}")
        #plt.savefig(f"output_epoch_{epoch+1}.png")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO2d(modes1=33, modes2=33, width=32).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
num_epochs = 100
train_model(model, train_loader, optimizer, criterion, num_epochs, device)



#
# Viz
#


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_rolling_prediction(model, dataloader, device, t_rolling, save_path="rolling_prediction.gif"):
    # Get a batch from the dataloader
    for batch in dataloader:
        break
        
    batch = batch.to('cuda:0')
    
    # Initialize the input with batch
    inputs = batch[-1:, :11, :, :, :]

    predictions = []

    # Collect predictions over rolling time steps
    for i in range(t_rolling):
        with torch.no_grad():
            logits = model(inputs[:, -6:])
            logits = logits.to(batch.device)
            logits = logits[:, -1:].unsqueeze(0)  # Use the last time step's prediction
            print(logits.shape)
        predictions.append(logits.cpu().numpy())

        # Use the predicted frame as the input for the next prediction
        inputs = torch.cat((inputs, logits), dim=1).detach()

    # Select a sample from the batch
    sample_index = 0  # Choose any index within the batch size
    

    # Extract the predicted values for the selected sample
    predicted_u = np.array([pred[sample_index, 0, 0, :, :] for pred in predictions])

    fig = plt.figure(figsize=(20, 15), dpi=72)
    moviewriter = animation.writers['pillow'](fps=4)  # Adjust FPS as needed
    moviewriter.setup(fig, save_path, dpi=72)

    def update(frame):
        plt.clf()
        gs = fig.add_gridspec(1, 1)

        ax_pred_u = fig.add_subplot(gs[0, 0])
        im_pred_u = ax_pred_u.imshow(predicted_u[frame], cmap='viridis')
        ax_pred_u.set_title('Predicted U-component')
        fig.colorbar(im_pred_u, ax=ax_pred_u)

        fig.suptitle(f"Time Step {frame + 1}/{t_rolling}")
        plt.draw()
        moviewriter.grab_frame()

    for i in range(t_rolling):
        update(i)

    moviewriter.finish()
    plt.close(fig)

plot_rolling_prediction(model, train_loader, device, t_rolling=1000, save_path="rolling_prediction_1000.gif")






