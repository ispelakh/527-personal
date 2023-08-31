from src import load_cifar10
if __name__ == '__main__':
    trainloader,testloader,classes = load_cifar10()
    print(len(trainloader))