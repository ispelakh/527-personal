from src import load_cifar10, Net, training_step,evaluate

if __name__ == '__main__':
    n_epochs = 10
    model = Net()
    # net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    trainloader,testloader,classes = load_cifar10()
    
    for epoch in range(n_epochs):
        training_step(model, trainloader, epoch)
        evaluate(model, testloader)
    print("-"*10,"Training finshed","-"*10)