import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

CONFIG = {
    "SEED": 1,
    "TRAIN_BATCH_SIZE": 64,
    "TEST_BATCH_SIZE": 1000,
    "EPOCHS": 15,
    "LEARNING_RATE": 1,
    "LOG_INTERVAL": 10,
}


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        relu = nn.ReLU()
        pool = nn.MaxPool2d(2)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5), relu, pool,  # C1 S2
            nn.Conv2d(6, 16, 5), relu, pool,  # C3 S4
            nn.Conv2d(16, 120, 5), relu,  # C5
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84), relu,  # F6
            nn.Linear(84, 10),  # F7
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)

        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch = batch_idx * len(data)
        total = len(train_loader.dataset)
        process_percentage = 100. * batch_idx / len(train_loader)
        loss_item = loss.item()
        total_loss += loss_item
        # if batch_idx % CONFIG["LOG_INTERVAL"] == 0:
        #     print(f"Train Epoch: {epoch} [{batch}/{total} ({process_percentage:.0f}%)]\tLoss: {loss_item:.6f}")

    average_loss = total_loss / len(train_loader)
    print(f"Train Epoch: {epoch}\tAverage Loss: {average_loss:.6f}")

    return average_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_num = 0
    test_num = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_num += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_num
    accuracy = correct_num / test_num * 100

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    return accuracy


def main():
    torch.manual_seed(CONFIG["SEED"])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": CONFIG["TRAIN_BATCH_SIZE"]}
    test_kwargs = {"batch_size": CONFIG["TEST_BATCH_SIZE"]}
    if use_cuda:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = LeNet5().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=CONFIG["LEARNING_RATE"])

    epochs = []
    train_loss_list = []
    test_accuracy_list = []
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        epochs.append(epoch)
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(test_accuracy)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('MNIST LeNet5 Result')

    ax1.plot(epochs, train_loss_list, 'r')
    ax1.set_ylabel('Train loss')

    ax2.plot(epochs, test_accuracy_list, 'b')
    ax2.axis([epochs[0], epochs[-1], 90, 100])
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Test accuracy')
    plt.show()


if __name__ == "__main__":
    main()
