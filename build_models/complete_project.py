#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print("Downloading pizza, steak, sushi data...")
  f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")


# In[2]:


os.makedirs("going_modular", exist_ok=True)


# In[3]:


get_ipython().run_cell_magic('writefile', 'going_modular/data_setup.py', '"""\nContains functionality for creating PyTorch DataLoaders for\nimage classification data.\n"""\nimport os\nfrom torchvision import datasets, transforms\nfrom torch.utils.data import DataLoader\n\nNUM_WORKERS = os.cpu_count()\n\ndef create_dataloaders(\n    train_dir: str,\n    test_dir: str,\n    train_transform: transforms.Compose,\n    test_transform: transforms.Compose,\n    batch_size: int,\n    num_workers: int=NUM_WORKERS\n):\n  """Creates training and testing DataLoaders.\n  """\n  # Use ImageFolder to create dataset(s)\n  train_data = datasets.ImageFolder(train_dir, transform=train_transform)\n  test_data = datasets.ImageFolder(test_dir, transform=test_transform)\n\n  # Get class names\n  class_names = train_data.classes\n\n  # Turn images into data loaders\n  train_dataloader = DataLoader(\n      train_data,\n      batch_size=batch_size,\n      shuffle=True,\n      num_workers=num_workers,\n      pin_memory=True,\n  )\n  test_dataloader = DataLoader(\n      test_data,\n      batch_size=batch_size,\n      shuffle=False, # don\'t need to shuffle test data\n      num_workers=num_workers,\n      pin_memory=True,\n  )\n\n  return train_dataloader, test_dataloader, class_names\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'going_modular/model_builder.py', '"""\nContains PyTorch model code to instantiate a TinyVGG model.\n"""\nimport os\nimport torch\nfrom torch import nn\n\nclass TinyVGG(nn.Module):\n  """Creates the TinyVGG architecture.\n  """\n  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:\n      super().__init__()\n      self.conv_block_1 = nn.Sequential(\n          nn.Conv2d(in_channels=input_shape,\n                    out_channels=hidden_units,\n                    kernel_size=3,\n                    stride=1,\n                    padding=0),\n          nn.ReLU(),\n          nn.Conv2d(in_channels=hidden_units,\n                    out_channels=hidden_units,\n                    kernel_size=3,\n                    stride=1,\n                    padding=0),\n          nn.ReLU(),\n          nn.MaxPool2d(kernel_size=2,\n                        stride=2)\n      )\n      self.conv_block_2 = nn.Sequential(\n          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n          nn.ReLU(),\n          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n          nn.ReLU(),\n          nn.MaxPool2d(2)\n      )\n      self.classifier = nn.Sequential(\n          nn.Flatten(),\n          # Where did this in_features shape come from?\n          # It\'s because each layer of our network compresses and changes the shape of our inputs data.\n          nn.Linear(in_features=hidden_units*13*13,\n                    out_features=output_shape)\n      )\n\n  def forward(self, x: torch.Tensor):\n      x = self.conv_block_1(x)\n      x = self.conv_block_2(x)\n      x = self.classifier(x)\n      return x\n      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion\n')


# In[17]:


get_ipython().run_cell_magic('writefile', 'going_modular/engine.py', '"""\nContains functions for training and testing a PyTorch model.\n"""\nimport torch\n\nfrom tqdm.auto import tqdm\nfrom typing import Dict, List, Tuple\n\ndef train_step(model: torch.nn.Module,\n               dataloader: torch.utils.data.DataLoader,\n               loss_fn: torch.nn.Module,\n               optimizer: torch.optim.Optimizer,\n               device: torch.device) -> Tuple[float, float]:\n  """Trains a PyTorch model for a single epoch.\n\n  """\n  # Put model in train mode\n  model.train()\n\n  # Setup train loss and train accuracy values\n  train_loss, train_acc = 0, 0\n\n  # Loop through data loader data batches\n  for batch, (X, y) in enumerate(dataloader):\n      # Send data to target device\n      X, y = X.to(device), y.to(device)\n\n      # 1. Forward pass\n      y_pred = model(X)\n\n      # 2. Calculate  and accumulate loss\n      loss = loss_fn(y_pred, y)\n      train_loss += loss.item()\n\n      # 3. Optimizer zero grad\n      optimizer.zero_grad()\n\n      # 4. Loss backward\n      loss.backward()\n\n      # 5. Optimizer step\n      optimizer.step()\n\n      # Calculate and accumulate accuracy metric across all batches\n      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)\n      train_acc += (y_pred_class == y).sum().item()/len(y_pred)\n\n  # Adjust metrics to get average loss and accuracy per batch\n  train_loss = train_loss / len(dataloader)\n  train_acc = train_acc / len(dataloader)\n  return train_loss, train_acc\n\ndef test_step(model: torch.nn.Module,\n              dataloader: torch.utils.data.DataLoader,\n              loss_fn: torch.nn.Module,\n              device: torch.device) -> Tuple[float, float]:\n  """Tests a PyTorch model for a single epoch.\n  """\n  # Put model in eval mode\n  model.eval()\n\n  # Setup test loss and test accuracy values\n  test_loss, test_acc = 0, 0\n\n  # Turn on inference context manager\n  with torch.inference_mode():\n      # Loop through DataLoader batches\n      for batch, (X, y) in enumerate(dataloader):\n          # Send data to target device\n          X, y = X.to(device), y.to(device)\n\n          # 1. Forward pass\n          test_pred_logits = model(X)\n\n          # 2. Calculate and accumulate loss\n          loss = loss_fn(test_pred_logits, y)\n          test_loss += loss.item()\n\n          # Calculate and accumulate accuracy\n          test_pred_labels = test_pred_logits.argmax(dim=1)\n          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))\n\n  # Adjust metrics to get average loss and accuracy per batch\n  test_loss = test_loss / len(dataloader)\n  test_acc = test_acc / len(dataloader)\n  return test_loss, test_acc\n\ndef train(model: torch.nn.Module,\n          train_dataloader: torch.utils.data.DataLoader,\n          test_dataloader: torch.utils.data.DataLoader,\n          optimizer: torch.optim.Optimizer,\n          loss_fn: torch.nn.Module,\n          epochs: int,\n          device: torch.device,\n          patience: int = 7,\n          name=None,\n           scheduler=None) -> Dict[str, List]:\n  """Trains and tests a PyTorch model with early stopping support."""\n\n  results = {\n      "train_loss": [],\n      "train_acc": [],\n      "test_loss": [],\n      "test_acc": []\n  }\n\n  best_loss = float("inf")\n  patience_counter = 0\n\n  for epoch in tqdm(range(epochs)):\n      train_loss, train_acc = train_step(\n          model=model,\n          dataloader=train_dataloader,\n          loss_fn=loss_fn,\n          optimizer=optimizer,\n          device=device\n      )\n\n      test_loss, test_acc = test_step(\n          model=model,\n          dataloader=test_dataloader,\n          loss_fn=loss_fn,\n          device=device\n      )\n\n      # Print epoch results\n      print(\n          f"Epoch: {epoch+1}/{epochs} | "\n          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "\n          f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f}"\n      )\n\n      # Save metrics\n      results["train_loss"].append(train_loss)\n      results["train_acc"].append(train_acc)\n      results["test_loss"].append(test_loss)\n      results["test_acc"].append(test_acc)\n      if scheduler:\n          scheduler.step(test_loss)\n\n      # --- Early Stopping Logic ---\n      if test_loss < best_loss:\n          best_loss = test_loss\n          patience_counter = 0\n          torch.save(model.state_dict(), "best_"+name+"_"+str(epoch)+".pth")\n          print("✅ Validation loss improved — best model saved.")\n      else:\n          patience_counter += 1\n          print(f"⚠️ No improvement for {patience_counter} epoch(s).")\n\n      if patience_counter >= patience:\n          print(f"\\n⏹️ Early stopping triggered at epoch {epoch+1}.")\n          break\n\n  print("\\nTraining complete.")\n  return results\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'going_modular/utils.py', '"""\nContains various utility functions for PyTorch model training and saving.\n"""\nimport torch\nfrom pathlib import Path\n\ndef save_model(model: torch.nn.Module,\n               target_dir: str,\n               model_name: str):\n  """Saves a PyTorch model to a target directory.\n  """\n  # Create target directory\n  target_dir_path = Path(target_dir)\n  target_dir_path.mkdir(parents=True,\n                        exist_ok=True)\n\n  # Create model save path\n  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with \'.pt\' or \'.pth\'"\n  model_save_path = target_dir_path / model_name\n\n  # Save the model state_dict()\n  print(f"[INFO] Saving model to: {model_save_path}")\n  torch.save(obj=model.state_dict(),\n             f=model_save_path)\n')


# In[7]:


get_ipython().run_cell_magic('writefile', 'going_modular/train.py', '"""\nTrains a PyTorch image classification model using device-agnostic code.\n"""\n\nimport os\nimport torch\nimport data_setup, engine, model_builder, utils\n\nfrom torchvision import transforms\n\n# Setup hyperparameters\nNUM_EPOCHS = 20\nBATCH_SIZE = 32\nHIDDEN_UNITS = 10\nLEARNING_RATE = 5e-4\nWEIGHT_DECAY=1e-4\n# Setup directories\ntrain_dir = "data/pizza_steak_sushi/train"\ntest_dir = "data/pizza_steak_sushi/test"\n\n# Setup target device\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\n\n# Create transforms\ntrain_transform = transforms.Compose([\n    transforms.Resize((64, 64)),\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomRotation(15),\n    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n    transforms.RandomAffine(0, translate=(0.1, 0.1)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\ntest_transform = transforms.Compose([\n    transforms.Resize((64, 64)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\n# Create DataLoaders with help from data_setup.py\ntrain_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(\n    train_dir=train_dir,\n    test_dir=test_dir,\n    train_transform=train_transform,\n    test_transform=test_transform,\n    batch_size=BATCH_SIZE\n)\n\n# Create model with help from model_builder.py\nmodel = model_builder.TinyVGG(\n    input_shape=3,\n    hidden_units=HIDDEN_UNITS,\n    output_shape=len(class_names)\n).to(device)\n\n# Set loss and optimizer\nloss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)\noptimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\nscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n    optimizer,\n    mode=\'min\',      # because we want to minimize validation loss\n    patience=2,      # wait 2 epochs before reducing LR\n    factor=0.5,      # multiply LR by 0.5 each time it plateaus\n)\n\n# Start training with help from engine.py\nengine.train(model=model,\n             train_dataloader=train_dataloader,\n             test_dataloader=test_dataloader,\n             loss_fn=loss_fn,\n             optimizer=optimizer,\n             epochs=NUM_EPOCHS,\n             device=device,\n             patience=7,\n             name="DuplicatedTinyVgg003",\n             scheduler=scheduler)\n\n# Save the model with help from utils.py\n# utils.save_model(model=model,\n#                  target_dir="models",\n#                  model_name="DuplicatedTinyVgg003.pth")\n')


# In[8]:


image_path


# In[9]:


get_ipython().system('python3 going_modular/train.py')


# In[10]:


#now with transfer learning
get_ipython().run_line_magic('%writefile', 'going_modular/googlenet003.py')




# In[18]:


get_ipython().run_cell_magic('writefile', 'going_modular/transfer_train.py', '"""\nTrains a PyTorch image classification model using device-agnostic code.\n"""\n\nimport os\nimport torch\nimport data_setup, engine\n\nfrom torchvision import transforms,models\n\nfrom torch import nn\n\n# Setup hyperparameters\nNUM_EPOCHS = 20\nBATCH_SIZE = 32\nHIDDEN_UNITS = 10\nLEARNING_RATE = 1e-3\nWEIGHT_DECAY=1e-4\n# Setup directories\ntrain_dir = "data/pizza_steak_sushi/train"\ntest_dir = "data/pizza_steak_sushi/test"\n\n# Setup target device\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\n\n# Create transforms\nfrom torchvision import transforms\n\ntrain_transform = transforms.Compose([\n    transforms.Resize((224, 224)),\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomRotation(15),\n    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n    transforms.RandomAffine(0, translate=(0.1, 0.1)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\ntest_transform = transforms.Compose([\n    transforms.Resize((224, 224)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\n\n# Create DataLoaders with help from data_setup.py\ntrain_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(\n    train_dir=train_dir,\n    test_dir=test_dir,\n    train_transform=train_transform,\n    test_transform=test_transform,\n    batch_size=BATCH_SIZE\n)\n\n# use googlenet pretrained model\ngooglenet_model = models.googlenet(weights=\'DEFAULT\')\nfor param in googlenet_model.parameters():\n  param.requires_grad = True\n\n# Replace the classifier head\nnum_features = googlenet_model.fc.in_features\ngooglenet_model.fc = nn.Linear(num_features, 3)\ngooglenet_model.to(device)\n\n# Set loss and optimizer\nloss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)\noptimizer = torch.optim.Adam(googlenet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n\nscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n    optimizer,\n    mode=\'min\',      # because we want to minimize validation loss\n    patience=2,      # wait 2 epochs before reducing LR\n    factor=0.5,      # multiply LR by 0.5 each time it plateaus\n)\n# Start training with help from engine.py\nengine.train(model=googlenet_model,\n             train_dataloader=train_dataloader,\n             test_dataloader=test_dataloader,\n             loss_fn=loss_fn,\n             optimizer=optimizer,\n             epochs=NUM_EPOCHS,\n             device=device,\n             patience=7,\n             name="googlenet",\n             scheduler=scheduler)\n\n# Save the model with help from utils.py\n# utils.save_model(model=model,\n#                  target_dir="models",\n#                  model_name="googlenet_003.pth")\n')


# In[19]:


get_ipython().system('python3 going_modular/transfer_train.py')


# In[9]:


get_ipython().run_cell_magic('writefile', 'going_modular/vision_transformer.py', 'import torch\nfrom torch import nn\nclass PatchEmbeding(nn.Module):\n    def __init__(self,\n                in_channels:int=3,\n                patch_size:int=16,\n                embed_dim:int=768):\n      super().__init__()\n      self.patch_size=patch_size\n      self.proj=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size,padding=0)\n      self.flatten=nn.Flatten(start_dim=2,end_dim=3)\n\n    def forward(self,x):\n      image_resolution=x.shape[-1]\n      assert image_resolution%self.patch_size==0,"image size must be divisible by patch size"\n      x=self.proj(x)\n      x=self.flatten(x)\n      return x.permute(0,2,1)\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self,\n                embed_dim:int=768, # Hidden size D from Table 1 for ViT-Base\n                num_heads:int=12, # Heads from Table 1 for ViT-Base\n                attn_dropout:float=0): # doesn\'t look like the paper uses any dropout in MSABlocks\n      super().__init__()\n      self.layer_norm=nn.LayerNorm(normalized_shape=embed_dim)\n      self.multi_head_attention=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout,batch_first=True)\n\n    def forward(self,x):\n      x=self.layer_norm(x)\n      atten_output,_=self.multi_head_attention(query=x,key=x,value=x,need_weights=False)\n      return atten_output\nclass MLP(nn.Module):\n  def __init__(self,\n              embed_dim:int=768, # Hidden Size D from Table 1 for ViT-Base\n              mlp_size:int=3072, # MLP size from Table 1 for ViT-Base\n              dropout:float=0.1): # Dropout from Table 3 for ViT-Base\n    super().__init__()\n    self.layer_norm=nn.LayerNorm(normalized_shape=embed_dim)\n    self.mlp=nn.Sequential(\n        nn.Linear(in_features=embed_dim,out_features=mlp_size),\n                nn.GELU(),\n        nn.Dropout(dropout),\n        nn.Linear(in_features=mlp_size,out_features=embed_dim),\n        nn.Dropout(p=dropout)\n    )\n  def forward(self,x):\n    x=self.layer_norm(x)\n    x=self.mlp(x)\n    return x\nclass TransformerBlock(nn.Module):\n    def __init__(self,\n                 embed_dim:int=768, # Hidden size D from Table 1 for ViT-Base\n                 num_heads:int=12, # Heads from Table 1 for ViT-Base\n                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base\n                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base\n                 attn_dropout:float=0): # Amount of dropout for attention layers\n        super().__init__()\n        self.msa_block=MultiHeadAttention(embed_dim=embed_dim,num_heads=num_heads,attn_dropout=attn_dropout)\n        self.mlp_block=MLP(embed_dim=embed_dim,mlp_size=mlp_size,dropout=mlp_dropout)\n    def forward(self,x):\n      x=self.msa_block(x)+x\n      x=self.mlp_block(x)+x\n      return x\n\nclass Vit(nn.Module):\n      def __init__(self,\n                 img_size:int=224, # Training resolution from Table 3 in ViT paper\n                 in_channels:int=3, # Number of channels in input image\n                 patch_size:int=16, # Patch size\n                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base\n                 embed_dim:int=768, # Hidden size D from Table 1 for ViT-Base\n                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base\n                 num_heads:int=12, # Heads from Table 1 for ViT-Base\n                 attn_dropout:float=0, # Dropout for attention projection\n                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers\n                 embed_dropout:float=0.1, # Dropout for patch and position embeddings\n                 num_classes:int=1000): # Default for ImageNet but can customize this\n          super().__init__() # don\'t forget the super().__init__()!\n          assert img_size % patch_size==0, "image size must be divisible by patch size"\n          num_patches=(img_size//patch_size)**2\n          self.patch_embeding=PatchEmbeding(in_channels=in_channels,patch_size=patch_size,embed_dim=embed_dim)\n          self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))\n          self.position_embed=nn.Parameter(torch.randn(1,1+num_patches,embed_dim))\n          self.embed_dropout=nn.Dropout(p=embed_dropout)\n          self.transform_encoder=nn.Sequential(*[\n              TransformerBlock(embed_dim,num_heads,mlp_size,mlp_dropout,attn_dropout) for _ in range(num_transformer_layers)\n          ])\n          self.classifier=nn.Sequential(\n              nn.LayerNorm(normalized_shape=embed_dim),\n              nn.Linear(in_features=embed_dim,out_features=num_classes)\n          )\n      def forward(self,x):\n        batch_size=x.shape[0]\n        x=self.patch_embeding(x)\n        cls_token=self.cls_token.expand(batch_size,-1,-1)\n        x=torch.cat((cls_token,x),dim=1)\n        x=x+self.position_embed\n        x=self.embed_dropout(x)\n        x=self.transform_encoder(x)\n        cls_token_final=x[:,0]\n        return self.classifier(cls_token_final)\n\n\n')


# In[13]:


get_ipython().run_cell_magic('writefile', 'going_modular/vit_train.py', '"""\nTrains a PyTorch image classification model using device-agnostic code.\n"""\n\nimport os\nimport torch\nimport data_setup, engine, vision_transformer\n\nfrom torchvision import transforms,models\n\nfrom torch import nn\n\n# Setup hyperparameters\nNUM_EPOCHS = 20\nBATCH_SIZE = 32\nHIDDEN_UNITS = 10\nLEARNING_RATE = 1e-3\nWEIGHT_DECAY=1e-4\n# Setup directories\ntrain_dir = "data/pizza_steak_sushi/train"\ntest_dir = "data/pizza_steak_sushi/test"\n\n# Setup target device\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\n\n# Create transforms\nfrom torchvision import transforms\n\ntrain_transform = transforms.Compose([\n    transforms.Resize((224, 224)),\n    transforms.RandomHorizontalFlip(),\n    transforms.RandomRotation(15),\n    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n    transforms.RandomAffine(0, translate=(0.1, 0.1)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\ntest_transform = transforms.Compose([\n    transforms.Resize((224, 224)),\n    transforms.ToTensor(),\n    transforms.Normalize([0.485, 0.456, 0.406],\n                         [0.229, 0.224, 0.225])\n])\n\n\n# Create DataLoaders with help from data_setup.py\ntrain_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(\n    train_dir=train_dir,\n    test_dir=test_dir,\n    train_transform=train_transform,\n    test_transform=test_transform,\n    batch_size=BATCH_SIZE\n)\n\n# use vit model\nmodel = vision_transformer.Vit(num_classes=len(class_names)).to(device)\n\n\n# Set loss and optimizer\nloss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)\noptimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n\nscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n    optimizer,\n    mode=\'min\',      # because we want to minimize validation loss\n    patience=2,      # wait 2 epochs before reducing LR\n    factor=0.5,      # multiply LR by 0.5 each time it plateaus\n)\n# Start training with help from engine.py\nengine.train(model=model,\n             train_dataloader=train_dataloader,\n             test_dataloader=test_dataloader,\n             loss_fn=loss_fn,\n             optimizer=optimizer,\n             epochs=NUM_EPOCHS,\n             device=device,\n             patience=7,\n             name="vit_003.pth",\n             scheduler=scheduler)\n\n# Save the model with help from utils.py\n# utils.save_model(model=model,\n#                  target_dir="models",\n#                  model_name="vit_003.pth")\n')


# In[14]:


get_ipython().system('python3 going_modular/vit_train.py')


# In[19]:


import torch
torch.cuda.is_available()


# In[20]:


torch.cuda.get_device_name()


# In[21]:


get_ipython().system('nvidia-smi')

