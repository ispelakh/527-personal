import torch, time
from src import load_cifar10, Net, ResNet50, ModelParallelResNet50,PipelineParallelResNet50, training_step,evaluate

if __name__ == '__main__':
    n_epochs = 10
    t1 = time.time()
    #model = ResNet50()
    #model = ModelParallelResNet50()
    #model = Net()
    model = PipelineParallelResNet50()
    
    net = torch.nn.DataParallel(model, device_ids=[0,1])
    
    # create a quantized model instance
    model_int8 = torch.ao.quantization.convert(model)

    trainloader,testloader,classes = load_cifar10()
    
    for epoch in range(n_epochs):
        # run quantized model_int8
        training_step(model_int8, trainloader, epoch)
        evaluate(model_int8, testloader)

        # normal model
        #training_step(model, trainloader, epoch)
        #evaluate(model, testloader)

    print("-"*10,"Training finished","-"*10)
    print(time.time()-t1)
