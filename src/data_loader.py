import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist():
    """Downloads and loads the MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_data_loaders(batch_size=64, val_split=0.2):
    """Splits the training set into training and validation loaders."""
    trainset, testset = load_mnist()
    train_size = int((1 - val_split) * len(trainset))
    val_size = len(trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
