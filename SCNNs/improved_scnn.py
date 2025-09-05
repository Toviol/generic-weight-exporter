# Imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configuração do dispositivo
if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Parâmetros globais
data_path = '/tmp/data/fashion_mnist'
spike_grad = surrogate.fast_sigmoid(slope=25)  # Gradiente para os spikes

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

# Arquitetura da SCNN Completa (Spiking CNN)
def create_scnn_complete(beta):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(16, 32, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(32, 64, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.4),

        nn.Flatten(),
        nn.Linear(64 * 3 * 3, 128),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.Dropout(0.4),
        nn.Linear(128, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)

# Forward pass temporal
def forward_pass(net, num_steps, data):
    spk_rec = []
    utils.reset(net)  # Reseta os estados dos neurônios
    for step in range(num_steps):  # Processamento ao longo do tempo
        spk_out, _ = net(data)
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

# Função de cálculo da acurácia
def batch_accuracy(loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec = forward_pass(net, num_steps, data)
            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)
    return acc / total

# Função de treinamento
def train_scnn(batch_size, beta, num_steps, num_epochs):
    train_loader = DataLoader(fashion_train_augmented, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False, drop_last=True)

    net = create_scnn_complete(beta)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = SF.ce_rate_loss()

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        epoch_acc = 0
        total_batches = len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            spk_rec = forward_pass(net, num_steps, data)
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Calculate batch accuracy
            batch_acc = SF.accuracy_rate(spk_rec, targets) * 100
            epoch_loss += loss_val.item()
            epoch_acc += batch_acc.item()

        # Scheduler step and epoch accuracy
        scheduler.step()
        epoch_loss /= total_batches
        epoch_acc /= total_batches

        print(f"Epoch {epoch + 1} Completed: Avg Loss = {epoch_loss:.4f}, Avg Accuracy = {epoch_acc:.2f}%\n")

    # Final test accuracy
    test_acc = batch_accuracy(test_loader, net, num_steps)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save the trained weights
    torch.save(net.state_dict(), "trained_scnn_fashion_mnist_weights_only.pt")
    print("Weights saved to: trained_scnn_fashion_mnist_weights_only.pt")
    
    return test_acc, net

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
beta = 0.7
num_steps = 100
num_epochs = 20

test_acc, trained_net = train_scnn(batch_size, beta, num_steps, num_epochs)

""" # Exportar dataset de teste
print("\n" + "="*50)
print("EXPORTANDO DATASET DE TESTE")
print("="*50)
export_test_dataset() """