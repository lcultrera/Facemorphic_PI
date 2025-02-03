import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM, CNNLSTM, CNNTransformerfEncoder, TransformerAlphaClassifier, LSTMAlphaClassifier, ResnetTwoStream, ViVit_Model
from facemorphic_dataset import FacemorphicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import bitsandbytes as bnb
import sys 
from vit_models.vivit_small_ds_both import ViT as VITSmall
import torch.nn as nn

params = {
    "mode": 'event_rgb',
    "learning_rate": 0.0001,
    "num_epochs": 10000,
    "batch_size_train": 2,
    "batch_size_test": 2,
    "max_seq_len": 50,
    "patch_size": 36,
    'au': 'AU',
    'weight_decay': 0.001,
    'best_model_path': "best_model.pth"
}
num_workers_test = 2


def test(model, dataloader_test, loss_fn, scheduler=None):
    model.eval()
    running_loss_au = 0.0
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    all_labels = []
    feats=[]

    for batch in tqdm(dataloader_test):
        if params['mode'] == 'alpha':
            inputs = batch['alpha'].cuda().float()
        if params['mode'] == 'event_rgb' or params["mode"]=='rgb_event':
            inputs_event = batch['event_imgs'].cuda().float()/255.0
            inputs_rgb = batch['rgb_imgs'].cuda().float()/255.0
        else:
            inputs = batch[f'{params["mode"]}_imgs'].cuda().float()/255.0
        labels = batch['label'].cuda()

        # Forward pass
        
        with torch.cuda.amp.autocast():
            if params['mode'] == 'alpha':
                outputs = model(inputs)
            if params['mode'] == 'event_rgb':
                rgb_feature,event_feature, original_feature,outputs = model(inputs_event.permute(0,1,4,2,3), inputs_rgb.permute(0,1,4,2,3))
            if params['mode'] == 'rgb_event':
                rgb_feature,event_feature,original_feature, outputs = model(inputs_rgb.permute(0,1,4,2,3), inputs_event.permute(0,1,4,2,3))
            #else:
                
                #outputs = model(inputs.permute(0,1,4,2,3))
            _, predicted = torch.max(outputs.data, 1)
            

            # Compute loss
            loss_au = loss_fn(outputs, labels)
        for b in range(rgb_feature.shape[0]):
            feats.append([rgb_feature[b].detach().cpu().numpy(),event_feature[b].detach().cpu().numpy(),original_feature[b].detach().cpu().numpy(),labels[b].detach().cpu().numpy()])
        np.save("features_E_AS_PI.npy",np.asarray(feats))
        # Update statistics
        running_loss_au += loss_au.item()
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    # Print statistics
    
    test_loss = running_loss_au / len(dataloader_test)
    if scheduler != None:
        scheduler.step(test_loss)
    test_accuracy = correct_predictions / total_predictions
    return test_loss, test_accuracy

model_event = VITSmall(
    image_size = 224,
    patch_size = 16,
    num_classes = 24,
    dim = 64,
    depth = 4,
    temporal_depth = 4,
    heads = 4,
    mlp_dim = 32,
    channels = 1,
    dropout = 0.1,
    emb_dropout = 0.1,
)

model_rgb2 = VITSmall(
    image_size = 224,
    patch_size = 16,
    num_classes = 24,
    dim = 64,
    depth = 4,
    temporal_depth = 4,
    heads = 4,
    mlp_dim = 32,
    channels = 3,
    dropout = 0.1,
    emb_dropout = 0.1,
)


model_rgb = VITSmall(
    image_size = 224,
    patch_size = 16,
    num_classes = 24,
    dim = 64,
    depth = 4,
    temporal_depth = 4,
    heads = 4,
    mlp_dim = 32,
    channels = 3,
    dropout = 0.1,
    emb_dropout = 0.1,
)

model_rgb.load_state_dict(torch.load('best_model-RGB_pretrained.pth'))
model_rgb2.load_state_dict(torch.load('best_model-RGB_pretrained.pth'))
#load model_event weights
model_event.load_state_dict(torch.load('best_model-event_pretrained.pth'))
#model_event2.load_state_dict(torch.load('best_model-event_pretrained.pth'))

# class CombinedModel(nn.Module):
#     def __init__(self, model_priviledged, model_original,model_hallucinated, num_classes):
#         super(CombinedModel, self).__init__()
#         self.priviledged_mod = model_priviledged
#         self.original_mod = model_original
#         self.hall_model = model_hallucinated
#         #i want freeze the event model
#         for param in self.priviledged_mod.parameters():
#             param.requires_grad = False
        
#         # The concatenated output size will be double the dimension of one model
        
#         combined_dim = 128 
        
#         # Define a fully connected layer to combine the outputs
#         self.fc = nn.Sequential(
#             nn.Linear(combined_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
#     def forward(self, priviledged_input, original_input):
#         # Forward pass through the event-based model
#         priviledged_output = self.priviledged_mod(priviledged_input)
        
#         # Forward pass through the RGB-based model
#         original_output = self.original_mod(original_input)
#         hall_output = self.hall_model(original_input)

#         # Concatenate the outputs along the last dimension
#         combined_output = torch.cat((original_output, hall_output), dim=-1)
        
#         # Pass through the fully connected layers
#         final_output = self.fc(combined_output)
        
#         return hall_output, priviledged_output,original_output, final_output
#i want a model tath use both model_rgb and model_event as sequential models
class CombinedModel(nn.Module):
    def __init__(self, model_event, model_rgb,model_rgb2, num_classes):
        super(CombinedModel, self).__init__()
        self.model_event = model_event
        self.model_rgb = model_rgb
        self.model_rgb2 = model_rgb2
        #i want freeze the event model
        for param in self.model_event.parameters():
            param.requires_grad = False
        
        # The concatenated output size will be double the dimension of one model
        
        combined_dim = 128 
        
        # Define a fully connected layer to combine the outputs
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    

    def forward(self, event_input, rgb_input):
        # Forward pass through the event-based model
        event_output = self.model_event(event_input)
        
        # Forward pass through the RGB-based model
        rgb_output = self.model_rgb(rgb_input)
        rgb_2_output = self.model_rgb2(rgb_input)

        # Concatenate the outputs along the last dimension
        combined_output = torch.cat((rgb_output, rgb_2_output), dim=-1)
        
        # Pass through the fully connected layers
        final_output = self.fc(combined_output)
        
        return rgb_2_output, event_output,rgb_output, final_output
# for p in model.resnet.parameters():
#     p.requires_grad = False

model = CombinedModel(model_event, model_rgb,model_rgb2, num_classes=24)
model.cuda()
model.load_state_dict(torch.load('best_models/EVENT_AS_PI.pth'))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')

data_transform = transforms.Compose([
       #transforms.RandomHorizontalFlip(),
       transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.NEAREST)
   ])

# test transform resize to 224
data_transform_test = transforms.Compose([
       transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
    ])
DATASET_PATH = '/andromeda/datasets/FACEMORPHIC/'
# data_transform = None
dataset_test = FacemorphicDataset(DATASET_PATH, split='test', mode=params['mode'], task=params['au'], toy=False, max_seq_len=params['max_seq_len'], transform=data_transform_test, use_annot=True, use_cache=True)
test_dataloader = DataLoader(dataset_test, batch_size=params['batch_size_test'], shuffle=False, num_workers=num_workers_test, drop_last=True)
loss_fn = torch.nn.CrossEntropyLoss()
test(model, test_dataloader,loss_fn=loss_fn)