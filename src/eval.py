import torch

def evaluate(model, test_loader):
    correct = 0
    total = 0
    device = 'cpu'
    if torch.cuda.is_available():
        #device = 'cuda:0'
        #device = 'cuda:1'
        device = 'cuda'
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            #outputs = model(images.to('cuda:0'))
            #outputs = model(images.to('cuda:1'))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return accuracy
