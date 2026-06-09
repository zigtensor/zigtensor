import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

class SimpleNet(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        x = torch.tanh(self.output(x))

        return x


def main():
    batch_size = 128
    learning_rate = 0.001
    epochs = 100
    input_size = 28 * 28
    hidden_size = 128
    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting the training run...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.5f}")

    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(train_loader))
        test_images = test_images.to(device)
        outputs = model(test_images[:5])

        predictions = torch.argmax(outputs, dim=1)

        print("\nSample predictions:")
        print("Predicted: ", predictions.cpu().tolist())
        print("Actual: ", test_labels[:5].tolist())

    print("\nModel Architecture:")
    print(model)

if __name__ == '__main__':
    main()




