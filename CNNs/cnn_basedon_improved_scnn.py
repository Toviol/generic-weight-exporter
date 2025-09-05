# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Configuração do dispositivo
if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Parâmetros globais
data_path = '/tmp/data/fashion_mnist'

# Transformação do conjunto de dados
transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Conjuntos de treino e teste
fashion_train_augmented = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform_augmented)
fashion_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform_augmented)

# Arquitetura da CNN equivalente à SCNN (sem spikes)
def create_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),  # 🔄 `snn.Leaky()` por `ReLU`

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),  # 🔄

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),  # 🔄

        nn.MaxPool2d(2),
        nn.Dropout(0.4),

        nn.Flatten(),
        nn.Linear(64 * 3 * 3, 128),
        nn.ReLU(),  # 🔄

        nn.Dropout(0.4),
        nn.Linear(128, 10)  # 🔄 Sem ativação "espinhosa" na saída
    ).to(device)

# Função de treinamento
def train_cnn(batch_size, num_epochs, learning_rate):
    train_loader = DataLoader(fashion_train_augmented, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False, drop_last=True)

    net = create_cnn()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()  # 🔄 Substituindo `SF.ce_rate_loss()`

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = net(data)  # 🔄 Apenas um único passo, sem spikes
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cálculo da acurácia
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            epoch_loss += loss.item()

        # Atualiza a taxa de aprendizado
        scheduler.step()
        epoch_loss /= total_batches
        acc = 100.0 * correct / total

        print(f"Epoch {epoch + 1} Completed: Avg Loss = {epoch_loss:.4f}, Accuracy = {acc:.2f}%\n")

    # Teste final
    test_acc = test_cnn(net, test_loader)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save the trained weights
    torch.save(net.state_dict(), "trained_cnn_fashion_mnist_weights_only.pt")
    print("Weights saved to: trained_cnn_fashion_mnist_weights_only.pt")
    
    return test_acc, net

# Função de teste
def test_cnn(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total

# Função para exportar dataset de teste
def export_test_dataset():
    """
    Exporta o conjunto de teste do FashionMNIST para arquivos .txt
    similar à implementação do N-MNIST
    """
    # Criar DataLoader para o conjunto de teste
    test_loader = DataLoader(fashion_test, batch_size=1, shuffle=True, drop_last=False)
    
    test_data = []
    test_targets = []
    
    print("Coletando dados do conjunto de teste...")
    for data, targets in test_loader:
        test_data.append(data)
        test_targets.append(targets)
    
    # Concatenar todos os dados
    test_data = torch.cat(test_data, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Reshape dos dados para formato apropriado
    # Para FashionMNIST: [batch_size, channels, height, width] -> [batch_size, height, width, channels]
    test_data = test_data.reshape(test_data.size(0), test_data.size(2), test_data.size(3), test_data.size(1)).type(torch.float)
    
    # Aplicar função sign para binarizar os dados (similar ao N-MNIST)
    test_data = torch.sign(test_data)
    
    print("Exportando dados para fashion-mnist_testset_data.txt...")
    # Exportar dados de teste
    with open('fashion-mnist_testset_data.txt', 'w') as f:
        for i in range(test_data.size(0)):
            for j in range(test_data.size(1)):
                for k in range(test_data.size(2)):
                    for l in range(test_data.size(3)):
                        f.write(str(int(test_data[i][j][k][l].item())))
                        f.write(' ')
                    f.write('\n')
    
    print("Exportando targets para fashion-mnist_testset_targets.txt...")
    # Exportar targets de teste
    with open('fashion-mnist_testset_targets.txt', 'w') as f:
        for i in range(test_targets.size(0)):
            f.write(str(test_targets[i].item()))
            f.write('\n')
    
    print(f"Dataset de teste exportado com sucesso!")
    print(f"Total de amostras: {test_data.size(0)}")
    print(f"Formato dos dados: {test_data.shape}")
    print(f"Arquivos criados:")
    print(f"  - fashion-mnist_testset_data.txt")
    print(f"  - fashion-mnist_testset_targets.txt")

# Executar o treinamento
batch_size = 128
num_epochs = 20
learning_rate = 1e-3

test_acc, trained_net = train_cnn(batch_size, num_epochs, learning_rate)

""" # Exportar dataset de teste
print("\n" + "="*50)
print("EXPORTANDO DATASET DE TESTE")
print("="*50)
export_test_dataset()
"""