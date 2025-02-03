import os
import subprocess

disalignments=[0,1,2,5,10]
#model_files = ['weights/late_fusion_pretrained_best_model-12575.pth','weights/early_fusion_pretrained_best_model-12575.pth','weights/middle_fusion_pretrained_best_model-12575.pth']
middle_fusion_model_files = ['/home/cioni/Facemorphic_PI/weights/early_fusion_best_model-2365335.pth']
for disalignment in disalignments:
    for model_file in middle_fusion_model_files:
        print(f"Disalignment: {disalignment}, Model: {model_file}")
        cmd = f"CUDA_VISIBLE_DEVICES='1'  /oblivion/users/lcultrera/anaconda3/envs/facemorphic/bin/python disalignment_test.py  {model_file} {disalignment}"
        print(cmd)
        os.system(cmd)
        print("Done")
    print("Done")