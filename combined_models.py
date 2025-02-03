import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import ModelEncoderDecoder, ModelEncoder, ResnetLSTM, CNNLSTM, CNNTransformerfEncoder, TransformerAlphaClassifier, LSTMAlphaClassifier, ResnetTwoStream, ViVit_Model


class CombinedModel_late_fusion(nn.Module):
    def __init__(self, model_priviledged, model_original,model_hallucinated, num_classes):
        super(CombinedModel_late_fusion, self).__init__()
        self.priviledged_mod = model_priviledged
        self.original_mod = model_original
        self.hall_model = model_hallucinated
        
        combined_dim = 128 
        
        # Define a fully connected layer to combine the outputs
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, priviledged_input, original_input):
        # Forward pass through the event-based model
        priviledged_output = self.priviledged_mod(priviledged_input)
        
        # Forward pass through the RGB-based model
        original_output = self.original_mod(original_input)
        #hall_output = self.hall_model(original_input)
        hall_output=None
        # Concatenate the outputs along the last dimension
        combined_output = torch.cat((original_output, priviledged_output), dim=-1)
        
        # Pass through the fully connected layers
        final_output = self.fc(combined_output)
        
        return hall_output, priviledged_output, final_output

class CombinedModel_early_fusion(nn.Module):
    def __init__(self, model_original, num_classes):
        super(CombinedModel_early_fusion, self).__init__()
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
        return None,None,final_output
    
class CombinedModel_middle_fusion(nn.Module):
    
    def __init__(self, backbone_event,backbone_rgb,temporal_encoder, num_classes):

        super(CombinedModel_middle_fusion, self).__init__()
        #self.priviledged_mod = model_priviledged
        self.backbone_event = backbone_event
        self.backbone_rgb = backbone_rgb
        self.temporal_encoder = temporal_encoder
        #self.hall_model = model_hallucinated
        #i want freeze the event model
        #for param in self.priviledged_mod.parameters():
        #    param.requires_grad = False
    

    def forward(self,rgb,event):
        rgb_output = self.backbone_rgb(rgb)
        event_output = self.backbone_event(event)
        print(rgb_output.shape,event_output.shape)
        combined = torch.cat([rgb_output,event_output],-1)
        final_output = self.temporal_encoder(combined)
        return None,None,final_output