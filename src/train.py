import torch, time
import torch.optim as optim
import torch.nn as nn
from dataset import get_cifar10
from model import LeNet5, VGGLike, ResNet18
from utils import plot_metrics

def train_model(model, trainloader, testloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss":[],"test_acc":[],"time":0}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss=0
        for inputs,labels in trainloader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history["train_loss"].append(running_loss/len(trainloader))

        # evaluate
        model.eval()
        correct,total=0,0
        with torch.no_grad():
            for inputs,labels in testloader:
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = model(inputs)
                _,predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        acc = 100*correct/total
        history["test_acc"].append(acc)
        print(f"Epoch {epoch+1}: Loss={running_loss:.3f}, Acc={acc:.2f}%")

    history["time"] = time.time()-start
    return history

if __name__=="__main__":
    trainloader,testloader = get_cifar10()
    models = {"LeNet5":LeNet5(),"VGGLike":VGGLike(),"ResNet18":ResNet18()}
    results={}
    for name,model in models.items():
        print(f"Training {name}...")
        results[name] = train_model(model,trainloader,testloader,epochs=5)
    plot_metrics(results)