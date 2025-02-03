from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

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


def debugger_is_active():
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

if debugger_is_active():
    print('DEBUGGER IS ACTIVE')
    num_workers_train = 1
    num_workers_test = 1
    comet_disabled = True
else:
    num_workers_train = 8
    num_workers_test = 2
    comet_disabled = False

experiment = Experiment(
    api_key="P2d8byK06cwHSJxxR9yXCkD5g",
    project_name="eccv-2024",
    workspace="facemorphic",
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    log_env_details=True,
    log_env_gpu=True,
    log_env_cpu=True,
    log_env_disk=True,
    disabled=comet_disabled
)

params = {
    "mode": 'rgb_event',
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
experiment.log_parameters(params)


# test function
def test(model, dataloader_test, loss_fn, scheduler=None):
    model.eval()
    running_loss_au = 0.0
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    all_labels = []

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
                outputs = model(inputs_event.permute(0,1,4,2,3), inputs_rgb.permute(0,1,4,2,3))
            if params['mode'] == 'rgb_event':
                outputs = model(inputs_rgb.permute(0,1,4,2,3), inputs_event.permute(0,1,4,2,3))
            else:
                outputs = model(inputs.permute(0,1,4,2,3))
            _, predicted = torch.max(outputs.data, 1)

            # Compute loss
            loss_au = loss_fn(outputs, labels)

        # Update statistics
        running_loss_au += loss_au.item()
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, predictions)
    experiment.log_confusion_matrix(matrix=cm, title="Confusion Matrix", file_name="Confusion Matrix.json", labels=[str(i) for i in range(24)])

    # Print statistics
    test_loss = running_loss_au / len(dataloader_test)
    if scheduler != None:
        scheduler.step(test_loss)
    test_accuracy = correct_predictions / total_predictions
    return test_loss, test_accuracy

# train loop
def train(model, dataloader_train, dataloader_test, loss_fn, loss_fn_mse, optimizer, num_epochs, scheduler=None):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss_au = 0.0
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0

        if epoch > 0:
            dataloader_train.clean_cache = False
            dataloader_test.clean_cache = False

        for n, batch in tqdm(enumerate(dataloader_train)):
            optimizer.zero_grad()

            if params['mode'] == 'alpha':
                inputs = batch['alpha'].cuda().float()

            if params['mode'] == 'event_rgb' or params["mode"]=='rgb_event':
                inputs_event = batch['event_imgs'].cuda().float()/255.0
                inputs_rgb = batch['rgb_imgs'].cuda().float()/255.0
            else:
                inputs = batch[f'{params["mode"]}_imgs'].cuda().float()/255.0
            labels = batch['label'].cuda()

            # if n == 0 and params['mode'] != 'alpha':
            #     frame_id = torch.argmax(torch.sum(inputs[0], dim=[1,2]))
            #     experiment.log_image(inputs[0, frame_id].detach().cpu().numpy(), name="input_frame")

            # Forward pass
            with torch.cuda.amp.autocast():
                if params['mode'] == 'alpha':
                    outputs = model(inputs)
                if params['mode'] == 'event_rgb':
                    outputs = model(inputs_event.permute(0,1,4,2,3), inputs_rgb.permute(0,1,4,2,3))
                if params['mode'] == 'rgb_event':
                    outputs = model(inputs_rgb.permute(0,1,4,2,3), inputs_event.permute(0,1,4,2,3))
                else:
                    outputs = model(inputs.permute(0,1,4,2,3))
                _, predicted = torch.max(outputs.data, 1)

                # Compute loss
                #i want an mse loss for rgb_feature with some gt fetures
                #loss_PI = loss_fn_mse(hall_feature, PI_feature) #loss between event and RGB
                loss_au = loss_fn(outputs, labels)
            total_loss = loss_au
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Update statistics
            running_loss_au += loss_au.item()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Print epoch statistics
        epoch_loss = running_loss_au / len(dataloader_train)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.4f}")
        experiment.log_metrics({"train_acc": accuracy}, epoch=epoch)

        # Test the model
        test_loss, test_accuracy = test(model, dataloader_test, loss_fn, scheduler)
        experiment.log_metrics({"test_acc": test_accuracy, "test_loss": test_loss}, epoch=epoch)
        print(f"Epoch {epoch}/{num_epochs}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            experiment.log_metric("best_acc", best_acc)
            torch.save(model.state_dict(), params['best_model_path'])
            experiment.log_asset(params['best_model_path'], file_name="best_model.pth")

# Create your model, dataset, loss function, and optimizer
#model = ModelEncoderDecoder(image_size=360, patch_size=params['patch_size'], seq_len=params['max_seq_len']) # TODO: fix num classes
#model = CNNLSTM(num_classes=24, hidden_dim=128, image_size=360)
#model = CNNTransformerfEncoder(num_classes=24, hidden_dim=128, image_size=360)
#model = ModelEncoder(image_size=360, patch_size=params['patch_size'])
#model = ResnetLSTM(image_size=224, hidden_dim=64, pretrain=True)
channels = 3
if params['mode'] == 'event':
    channels = 1
#model = ViVit_Model(image_size=224, frames=params['max_seq_len'], image_patch_size=16, frame_patch_size=1, num_classes=24, dim=64, spatial_depth=4, temporal_depth=4, heads=4, mlp_dim=32, channels=channels)
# model = ViVit_Model(image_size=224,
#                     frames=50,
#                     image_patch_size=16,
#                     frame_patch_size=1,
#                     num_classes=24,
#                     dim=1024,
#                     spatial_depth=2,
#                     temporal_depth=2,
#                     heads=4,
#                     mlp_dim=512,
#                     channels=1)
#model = LSTMAlphaClassifier()
#model = ResnetTwoStream(image_size=224, seq_len=params['max_seq_len'], feat_dim=64, feat_dim2= 32, hidden_dim=256, mlp_dim=128)
# model_event = VITSmall(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 24,
#     dim = 64,
#     depth = 4,
#     temporal_depth = 4,
#     heads = 4,
#     mlp_dim = 32,
#     channels = 1,
#     dropout = 0.1,
#     emb_dropout = 0.1,
# )

# model_event2 = VITSmall(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 24,
#     dim = 64,
#     depth = 4,
#     temporal_depth = 4,
#     heads = 4,
#     mlp_dim = 32,
#     channels = 1,
#     dropout = 0.1,
#     emb_dropout = 0.1,
# )


model_combined = VITSmall(
    image_size = 224,
    patch_size = 16,
    num_classes = 24,
    dim = 64,
    depth = 4,
    temporal_depth = 4,
    heads = 4,
    mlp_dim = 32,
    channels = 4,
    dropout = 0.1,
    emb_dropout = 0.1,
)

# model_rgb2 = VITSmall(
#     image_size = 224,
#     patch_size = 16,
#     num_classes = 24,
#     dim = 64,
#     depth = 4,
#     temporal_depth = 4,
#     heads = 4,
#     mlp_dim = 32,
#     channels = 3,
#     dropout = 0.1,
#     emb_dropout = 0.1,
# )

#load model_rgb weights
#model_rgb.load_state_dict(torch.load('best_model-RGB_pretrained.pth'))
#model_rgb2.load_state_dict(torch.load('best_model-RGB_pretrained.pth'))
#load model_event weights
#model_event.load_state_dict(torch.load('best_model-event_pretrained.pth'))
#model_event2.load_state_dict(torch.load('best_model-event_pretrained.pth'))

class CombinedModel(nn.Module):
    def __init__(self, model_original, num_classes):
        super(CombinedModel, self).__init__()
        #self.priviledged_mod = model_priviledged
        self.original_mod = model_original
        #self.hall_model = model_hallucinated
        #i want freeze the event model
        #for param in self.priviledged_mod.parameters():
        #    param.requires_grad = False
    

    def forward(self, priviledged_input, original_input):
        # Forward pass through the event-based model
        #priviledged_output = self.priviledged_mod(priviledged_input)
        combined = torch.cat((original_input, priviledged_input), dim=2) 
        # Forward pass through the RGB-based model
        original_output = self.original_mod(combined)
        #hall_output = self.hall_model(original_input)

        # Concatenate the outputs along the last dimension
        #combined_output = torch.cat((original_output, hall_output), dim=-1)
        
        # Pass through the fully connected layers
        final_output = original_output#self.fc(combined_output)
        return final_output
        #return hall_output, priviledged_output, final_output
#i want a model tath use both model_rgb and model_event as sequential models

# for p in model.resnet.parameters():
#     p.requires_grad = False

model = CombinedModel(model_combined, num_classes=24)
model.cuda()
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
dataset_train = FacemorphicDataset(DATASET_PATH, split='train', mode=params['mode'], task=params['au'], toy=False, max_seq_len=params['max_seq_len'], transform=data_transform, use_annot=True, use_cache=True)
dataset_test = FacemorphicDataset(DATASET_PATH, split='test', mode=params['mode'], task=params['au'], toy=False, max_seq_len=params['max_seq_len'], transform=data_transform_test, use_annot=True, use_cache=True)

loss_fn = torch.nn.CrossEntropyLoss()
# define mse loss
loss_fn_mse = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
scheduler = None # ReduceLROnPlateau(optimizer, 'min')
# optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=params['learning_rate']) # instead of torch.optim.Adam

# dataloader
train_dataloader = DataLoader(dataset_train, batch_size=params['batch_size_train'], shuffle=True, num_workers=num_workers_train, drop_last=True)
test_dataloader = DataLoader(dataset_test, batch_size=params['batch_size_test'], shuffle=False, num_workers=num_workers_test, drop_last=True)

# Set the number of epochs and start training
train(model, train_dataloader, test_dataloader, loss_fn, loss_fn_mse, optimizer, params['num_epochs'], scheduler)

