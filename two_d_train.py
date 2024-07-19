import os
from tqdm import tqdm
import time
import datetime
from datetime import datetime

from torch.utils.data import  DataLoader
import torch
import torch.optim as optim
import wandb

from two_d_data_set import  SonarDataset
from utils import  extract_path_from_dir
from constants import Constants
from packages.schedualer import LRScheduler, SaveBestModel
from utils import norms
from two_d_model import deeponet

if __name__=="__main__":
    n=Constants.n
    model=deeponet(dim=2,f_shape=n**2, domain_shape=2, p=80) 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    
    train_data=extract_path_from_dir(Constants.train_path+'data/')
    train_masks=extract_path_from_dir(Constants.train_path+'masks/')
    train_domains=extract_path_from_dir(Constants.train_path+'domains/')
    train_functions=extract_path_from_dir(Constants.train_path+'functions/')
    
    s_train=[torch.load(f) for f in train_data]
    masks_train=torch.load(train_masks[0])
    domains_train=torch.load(train_domains[0])
    functions_train=torch.load(train_functions[0])
    
    X_train=[]
    Y_train=[]
    for s in s_train:
        X=s[0]
        y=X[0]
        mask= masks_train[X[2]]
        domain= domains_train[X[2]]
        function=functions_train[X[2]][X[1]]
        X_train.append([y,function,domain,mask])
        Y_train.append(s[1])
    train_dataset = SonarDataset(X_train, Y_train)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False,num_workers=0)
    val_dataloader=DataLoader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False,num_workers=0)
        
    test_data=extract_path_from_dir(Constants.test_path+'data/')
    test_masks=extract_path_from_dir(Constants.test_path+'masks/')
    test_domains=extract_path_from_dir(Constants.test_path+'domains/')
    test_functions=extract_path_from_dir(Constants.test_path+'functions/')
    
    s_test=[torch.load(f) for f in test_data]
    masks_test=torch.load(test_masks[0])
    domains_test=torch.load(test_domains[0])
    functions_test=torch.load(test_functions[0])
    
    X_test=[]
    Y_test=[]
    for s in s_test:
        X=s[0]
        y=X[0]
        mask= masks_test[X[2]]
        domain= domains_test[X[2]]
        function=functions_test[X[2]][X[1]]
        X_test.append([y,function,domain,mask])
        Y_test.append(s[1])
    
    test_dataset = SonarDataset(X_test, Y_test)
    test_dataloader=DataLoader(test_dataset, batch_size=Constants.batch_size, shuffle=False, drop_last=False)



    # lsof -i:6006 
    # kill -9 68614
    # tensorboard --logdir=/Users/idanversano/Documents/project_geo_deeponet/two_d_lshape/runs/

    experment_path=Constants.path+'runs/'+datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    isExist = os.path.exists(experment_path)
    if not isExist:
        os.makedirs(experment_path)  
# 2024.06.04.22.02.41best_model.pth

    # best_model=torch.load(Constants.path+'runs/'+'2024.07.03.14.35.38best_model.pth')
    # model.load_state_dict(best_model['model_state_dict'])
    
    wandb.init(
    project="deeponet",
    config={
    "learning_rate": 1e-4,
    "architecture": "deeponet",
    "dataset": "L-shpae,8, subsquares",
    "epochs": 6000,
    }
)


    lr = 0.0001
    epochs = Constants.num_epochs
    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-5)
    criterion = norms.relative_L2

    # scheduler
    lr_scheduler=LRScheduler(optimizer)
    # early_stopping = EarlyStopping()
    save_best_model = SaveBestModel(experment_path)
    # device
    device = Constants.device

    def fit(model, train_dataloader, train_dataset, optimizer, criterion):
        train_running_loss = 0.0
        counter = 0
        total = 0
        prog_bar = tqdm(
            enumerate(train_dataloader),
            total=int(len(train_dataset) / train_dataloader.batch_size),
        )
        for i, data in prog_bar:
                counter += 1
                inp, output = data
                inp = [inp[k].to(Constants.device) for k in range(len(inp))]
                output = output.to(Constants.device) 
                total += output.size(0)
                optimizer.zero_grad()
                y_pred = model(inp)
            
                loss = criterion(y_pred, output) 
                train_running_loss += loss.item()

                loss.backward()
                optimizer.step()

        train_loss = train_running_loss / counter
        return train_loss

    
    def predict(model, dataloader, dataset, criterion):
        pred_running_loss = 0.0
        counter = 0
        total = 0
        prog_bar = tqdm(
            enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
        )
        with torch.no_grad():
            for i, data in prog_bar:
                counter += 1
                inp, output = data
                inp = [inp[k].to(Constants.device) for k in range(len(inp))]
                output = output.to(Constants.device)
                total += output.size(0)
                y_pred = model(inp)
            
                loss = criterion(y_pred, output) 
                pred_running_loss += loss.item()

            pred_loss = pred_running_loss/counter
            return pred_loss
        
    def validate(model, dataloader, dataset, criterion):
        val_running_loss = 0.0
        counter = 0
        total = 0
        prog_bar = tqdm(
            enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
        )

        with torch.no_grad():
            for i, data in prog_bar:
                counter += 1
                inp, output = data
                inp = [inp[k].to(Constants.device) for k in range(len(inp))]
                output = output.to(Constants.device) 
                total += output.size(0)
                y_pred = model(inp)
                
                loss = criterion(y_pred, output)

                val_running_loss += loss.item()

            val_loss = val_running_loss / counter
            return val_loss


    train_loss=[]
    val_loss=[]
    test_loss=[]
    start = time.time()

    model.to(Constants.device)
    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1} of {epochs}")
        
        train_epoch_loss= fit(model, train_dataloader, train_dataset, optimizer, criterion)
        val_epoch_loss = validate(model, val_dataloader, val_dataset, criterion)
        test_epoch_loss  = predict(model, test_dataloader, test_dataset, criterion)

        lr_scheduler(val_epoch_loss)
        save_best_model(val_epoch_loss, epoch, model, optimizer, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        test_loss.append(test_epoch_loss)

        print("-" * 50)
        print(f"Train Loss: {train_epoch_loss:4e}")
        print(f"Val Loss: {val_epoch_loss:.4e}")
        print(f"Test Loss: {test_epoch_loss:.4e}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.4e}")
        
        wandb.log({"train_loss":train_epoch_loss,
                   "val_loss":val_epoch_loss,
                   "test_loss":test_epoch_loss,
                   "lr": optimizer.param_groups[0]['lr']})
    end = time.time()
    print(f"Training time: {(end-start)/60:.3f} minutes")

    try:
        wandb.finish()

    except:
        pass    

