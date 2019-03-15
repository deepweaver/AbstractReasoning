import argparse 
import torch 
import torchvision 
import os, sys
from model import *
# from Preprocess import *
from utils import *
import matplotlib.pyplot as plt 
if not os.path.isdir("./results"):
    os.mkdir('./results') 
if not os.path.isdir('./plots'):
    os.mkdir('./plots')

def train(args, model, device, train_loader, optimizer, epoch, train_loss):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)
        # print(target.shape)
    # torch.Size([128, 9, 80, 80])
    # torch.Size([128])
        # model(data)
        # print(type(data))
        # print(type(target))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        # print(type(output))
        loss = torch.nn.CrossEntropyLoss()
        # print(type(output))
        # print(type(target))
        ls = loss(output, target) 
        # train_loss
        # print(ls.data.cpu().numpy())
        ls.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            train_loss.append(ls.data.cpu().numpy())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data),len(train_loader.dataset),100.0 * batch_idx / len(train_loader), ls.data.cpu().numpy()))
        # break 
def test(args, model, device, test_loader, test_accuracy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        x = 0
        for _, (data, target) in enumerate(test_loader):
            x += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()
            ls = loss(output, target)
            test_loss += ls.data.cpu().numpy()
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            # if x == 3:
            #     break
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print("Test set average loss = {:.4f}, accuracy = {}/{} ({:.0f}%)".format(
            test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
           accuracy)) 
    test_accuracy.append(accuracy) 






def main():

    parser = argparse.ArgumentParser(description='reasoning test')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False) 
    parser.add_argument('--seed', type=int, default=1, metavar='S') 
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging traning status')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='/Users/tiany/Downloads/novel.domain.transfer/')
    parser.add_argument('--save-model', action='store_true', default=True) 
    parser.add_argument('--model', type=str, default='./saved_models/CRModel.pt')
    parser.add_argument('--only-test', action='store_true', default=False) # default is train + test
    args = parser.parse_args() 
    args.cuda = not args.no_cuda and torch.cuda.is_available() 
    print("Using cuda is {}".format(args.cuda)) 
    torch.manual_seed(args.seed) 
    np.random.seed(args.seed) 
    device = torch.device('cuda' if args.cuda else 'cpu') 
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {} 
    kwargs.update({'batch_size': args.batch_size, 'shuffle': True})
    train_loader = torch.utils.data.DataLoader(
        dataset=NormalDataset(
            root=args.data_path,
            train=True,
            transform=torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=NormalDataset(
            root=args.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        **kwargs
    )
    
    
    if args.only_test:
        # pass
        model = torch.load(args.model)
        # model.load_state_dict(torch.load(args.model)) 
        # model.eval()
        test_accuracy = []
        test(args, model, device, test_loader, test_accuracy) 

    else:
        model = CRModel().to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


        train_loss = []
        test_accuracy = [] 
        for epoch in range(1, args.epochs+1):
            train(args, model, device, train_loader, optimizer, epoch, train_loss)
            test(args, model, device, test_loader, test_accuracy) 
            # break 
        
        plt.plot(train_loss)
        plt.savefig('./plots/train_loss.png')
        plt.cla()
        plt.plot(test_accuracy)  
        plt.savefig('./plots/test_accuracy.png')


        if args.save_model:
            torch.save(model, './saved_models/CRModel.pt')
if __name__ == "__main__":
    main()