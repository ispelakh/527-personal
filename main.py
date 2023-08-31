from src import load_cifar10, Net, training_step

if __name__ == '__main__':
    model = Net()
    # print(model)
    trainloader,testloader,classes = load_cifar10()
    training_step(model, trainloader, 0)