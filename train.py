import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Tintelligence
from components.dataset import train_dataloaded
from tqdm import tqdm
import os
import torchvision.utils as vutils
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


output_path="content"
criterion = nn.CrossEntropyLoss()
model=Tintelligence()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
scheduler =  torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2,step_size_up=5,mode="exp_range",gamma=0.85)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()

def save_generated_images(model, epoch, device, output_path, num_samples=16):
    model.eval()
    with torch.no_grad():
        # Rastgele birkaç gerçek resimle t oluştur
        sample_images = next(iter(train_dataloaded))[0:num_samples].to(device)
        fake_images=model(sample_images)

        os.makedirs(output_path, exist_ok=True)
        vutils.save_image(
            fake_images,
            os.path.join(output_path, f"fake_epoch_{epoch+1}.png"),
            normalize=True,
            nrow=int(num_samples**0.5),
            value_range=(0, 1),
        )
    model.train()

def train(model,trainloader:torch.utils.data.DataLoader,epochs:int,grad_accum=4):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total=0
        progress_bar = tqdm(trainloader, total=len(trainloader), 
                            desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for i,(gray,target) in enumerate(progress_bar):
            
            gray, target = gray.to(device), target.to(device)

            # zero the parameter gradients
            
            with torch.cuda.amp.autocast():
                
                outputs = model(gray)
                
                loss = criterion(outputs, target)
            running_loss += loss.item()
            loss=loss/grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(trainloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total += target.size(0)
            progress_bar.set_postfix(loss=f"{running_loss / (total / 1):.4f}")
        print('Loss: {}'.format(running_loss/len(trainloader)))
        #save_generated_images(model, epoch, device, output_path, 16)
    print('Finished Training')

train(model,train_dataloaded,100)

