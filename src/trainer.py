import torch 
import time 

def concrete_trainer(net, loss, dataloader, num_epoch, learning_rate, weight_decay, device, init_temp= 10, end_temp= 0.1):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)
    start_time  = time.time()
    net.train()
    for epoch in range(1, num_epoch+1):
        train_loss = 0
        temp = init_temp * (end_temp/init_temp)**(epoch/num_epoch)
        for data in dataloader:
            input = data.to(device, non_blocking = True)
            optimizer.zero_grad()
            reconstruction, _, _ = net(input, temp)
            l = loss(reconstruction, input)
            l.backward()
            optimizer.step()
            train_loss += l
        process_time = time.time() - start_time
        train_loss /= len(dataloader)
        print("Epoch: %d ; Loss: %.5f; Temperature: %.3f; Time: %.2f s" %(epoch, train_loss,temp, process_time))
