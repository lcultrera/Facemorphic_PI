from regex import W
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM, CNNLSTM, CNNTransformerfEncoder, TransformerAlphaClassifier, LSTMAlphaClassifier, ResnetTwoStream, ViVit_Model
from facemorphic_dataset_testing import FacemorphicDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import bitsandbytes as bnb
import sys 
from vit_models.vivit_small_ds_both import ViT as VITSmall
#from vit_models.vivit_small_middle_fusion import ViT as VITSmall
#from vit_models.vivit_small_middle_fusion_temporal import ViT as ViTTemporal
import torch.nn as nn
import combined_models as cm
import sys

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
                print(inputs_event.shape,inputs_rgb.shape)
                rgb_feature,event_feature, outputs = model(inputs_event.permute(0,1,4,2,3), inputs_rgb.permute(0,1,4,2,3))
            if params['mode'] == 'rgb_event':
                print(inputs_event.shape,inputs_rgb.shape,labels.shape)
                rgb_feature,event_feature, outputs = model(inputs_rgb.permute(0,1,4,2,3), inputs_event.permute(0,1,4,2,3))
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
    #experiment.log_confusion_matrix(matrix=cm, title="Confusion Matrix", file_name="Confusion Matrix.json", labels=[str(i) for i in range(24)])

    # Print statistics
    test_loss = running_loss_au / len(dataloader_test)
    if scheduler != None:
        scheduler.step(test_loss)
    test_accuracy = correct_predictions / total_predictions
    return test_loss, test_accuracy


DISALIGNMENT=int(sys.argv[2])
WEIGHTS_FILE=sys.argv[1]
DATASET_PATH = '/andromeda/datasets/FACEMORPHIC/'
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
    'best_model_path': "best_model.pth",
    'disaligned': "rgb",
    'disalignment_max': DISALIGNMENT
}
data_transform_test = transforms.Compose([
       transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
    ])
dataset_test = FacemorphicDataset(DATASET_PATH, split='test', mode=params['mode'], task=params['au'], 
                                  toy=False, max_seq_len=params['max_seq_len'], transform=data_transform_test, 
                                  use_annot=True, use_cache=True,disaligned=params['disaligned'],disalignment_max=params['disalignment_max'])

# dt = next(iter(dataset_test))
# for k in dt:
#     print(k,dt[k].shape)
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

model_event2 = VITSmall(
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


model_rgb = VITSmall(
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

#model = cm.CombinedModel_late_fusion(model_rgb, model_event,model_event2, num_classes=24)
#weights_file = 'weights/late_fusion_pretrained_best_model-12575.pth'
model = cm.CombinedModel_early_fusion(model_rgb, num_classes=24)
#model = cm.CombinedModel_middle_fusion(model_event, model_event2, model_rgb, num_classes=24)
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()

test_loader = DataLoader(dataset_test, batch_size=params['batch_size_test'], shuffle=False, num_workers=4)
test_loss,test_accuracy=test(model, test_loader, loss_fn, scheduler=None)
print(f"Disalignment {DISALIGNMENT}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
with open(f"disalignment_test_rgb.txt",'a') as f:
    f.write(f"Weightsfile : {WEIGHTS_FILE}, Disalignment :{DISALIGNMENT}, Test Loss : {test_loss:.4f}, Test Accuracy : {test_accuracy:.4f}\n")
f.close()
