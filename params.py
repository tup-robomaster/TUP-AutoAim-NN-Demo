import yaml
import numpy as np

class CamParam:
    cam_mat = []
    dis_coeff = []
    def __init__(self):
        #Init
        path = "/home/rangeronmars/RM/Demo/sjtu_autoaim_pytorch/params/camera_param.yml"
        #Load
        with open(path,"r") as cam_param_file:
            cam_param_content = cam_param_file.read()
            cam_param = yaml.load(cam_param_content, Loader=yaml.FullLoader)
            cam_mat = np.array(cam_param["Camera_Matrix"]).reshape(3,3)
            dis_coeff = np.array(cam_param["Distortion_Coefficients"])
            self.cam_mat = cam_mat
            self.dis_coeff = dis_coeff

class Armor:
    armor_size_small = np.array([13, 5.5])  # Width,Height(cm)
    armor_size_big = np.array([22.5, 5.5])  # Width,Height(cm)
    #def __init__(self):

