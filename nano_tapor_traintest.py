# simpler edge tapor model

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
import pickle   
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
from dataset import iHand_dataset, RandomOrthogonalRotation
import time
from tapor_model_config import small_setting, base1_setting, base2_setting, large1_setting, large2_setting, base1_setting_varaint1, base1_setting_varaint2
from models import BlazeHandLandmark, ColorHandPose3D, Mano_MobileNetV2, Tapor, TaporTeacher
from torch.utils.tensorboard import SummaryWriter
from utils import draw_landmarks, draw_3D_landmarks, generate_heatmaps, calculate_mpjpe, calculate_pck, calculate_shift_error,BoneLengthLoss, KinematicChainLoss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class EdgeTaporStudent(nn.Module):
    def __init__(self, feat_dim =21*48, last_conv_channel=128):
        super(EdgeTaporStudent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, last_conv_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(last_conv_channel*3*4, feat_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 21*3)
        )
    
    def forward(self, thermal_map):
        feat = self.encoder(thermal_map)
        pose = self.decoder(feat)
        return pose, feat
 
# define a model with only one fully connected layer
class Adaptor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Adaptor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class KD_edgeTapor_dataset(Dataset):
    def __init__(self, datapath, feature_type = 0, feature_dim = 21*20):
        '''ArithmeticError
        Args:
            datapath: the path of the dataset for knowledge distillation training
            feature_type:
                            0: no KD distillation, 
                            1 for cross-attention features, 
                            2 for slef-attention features
            feature_dim: the dimension of the feature after PCA
        '''
        try:
            with open(datapath, 'rb') as f:
                self.thermal_maps, self.labels, self.cross_features, self.self_features = pickle.load(f)
        except:
            print("The knowledge distillation training data file does not exist!")
        self.feature_type = feature_type
        if feature_dim == 21*48:
            pass   # now, we use the adaptor to change the feature dimention, not the PCA inside the class
        else:
            pca = PCA(n_components=feature_dim)
            if self.feature_type == 0:
                pass
            elif self.feature_type == 1:
                self.cross_features = pca.fit_transform(self.cross_features) 
            elif self.feature_type == 2:
                self.self_features = pca.fit_transform(self.self_features) 

    def __len__(self):
        return len(self.self_features)

    def __getitem__(self, idx):
        if self.feature_type == 0:
            return self.thermal_maps[idx], self.labels[idx], 0.
        elif self.feature_type == 1:
            return self.thermal_maps[idx], self.labels[idx], self.cross_features[idx]
        elif self.feature_type == 2:
            return self.thermal_maps[idx], self.labels[idx], self.self_features[idx]
        return self.thermal_maps[idx], self.labels[idx], 0.
            

def inference(model,data_loader,device):
    model.eval()  # Set the model to evaluation mode
    predicts_all = {
        'joints_3d': [],
        'feat': [],
    }
    labels_all = {
        'thermal_map': [],
        'joints_3d': [],
    }
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, input in enumerate(tqdm(data_loader)):
            thermal_map, ambient_temperature, l_depth_map, l_2d_joint, l_2d_flag, l_3d_joint,\
                l_3d_flag, l_hand_depth, l_left_right_flag, heatmap_label = input
            label = l_3d_joint.squeeze().float().to(device)
            thermal_map = thermal_map.float().to(device)
            landmarks, feat = model(thermal_map)
            landmarks = landmarks.view(-1, 21, 3)
            predicts_all['joints_3d'].append(landmarks.cpu().numpy())
            predicts_all['feat'].append(feat.cpu().numpy())
            labels_all['thermal_map'].append(thermal_map.cpu().numpy())
            labels_all['joints_3d'].append(label.cpu().numpy())
    return predicts_all,labels_all


def draw_3D_landmarks2(thermal_map,prediction,groundtruth =None):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    height_default, dpi_default= (4.8, 300)
    # height,wide = thermal_map.shape
    height,wide = (256*3,256*3)

    lines = [[0,1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20], [0,5,9,13,17,0]]
    for line in lines:
        x,y,z = [],[],[]
        label_x,label_y,label_z = [],[],[]
        for id in line:
            x.append(-prediction[id][0])
            y.append(-prediction[id][1])
            z.append(prediction[id][2])
            if groundtruth is not None:
                label_x.append(-groundtruth[id][0])
                label_y.append(-groundtruth[id][1])
                label_z.append(groundtruth[id][2])
        ax.plot(x,z,y,color='g', linestyle="-", linewidth=1)
        ax.scatter(x, z, y, s=10, c='r', marker='o')
        if groundtruth is not None:
            ax.plot(label_x,label_z,label_y,color='k', linestyle="-", linewidth=1)
            ax.scatter(label_x, label_z, label_y, s=10, c='b', marker='o')

    # set the limits of the plot to the limits of the data
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 0.6)
    ax.set_zlim(-0.2, 0.2)
    # set the stride of the ticks
    ax.xaxis.set_ticks(np.arange(-0.2, 0.21, 0.1))
    ax.yaxis.set_ticks(np.arange(0, 0.61, 0.1))
    ax.zaxis.set_ticks(np.arange(-0.2, 0.21, 0.1))
    
    fig.canvas.draw()
    fig_str = fig.canvas.tostring_rgb()
    
    plot_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(height_default*dpi_default), -1, 3)) #/ 255.0
    plot_img = cv2.resize(plot_img,(wide*3,height*3))
    plt.close(fig)
    return plot_img


def visualization_results_video(log_folder, log_file_name,input_thermal_frames, groundtruth_3d_joints, estimated_3d_joints, t_size = (256*3,256*3),file_names_for_visualization = None):
    # input_thermal_frames: (n, 256, 256, 3)
    # groundtruth_3d_joints: (n, 21, 3)
    # estimated_3d_joints: (n, 21, 3)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height,wide = t_size
    out = cv2.VideoWriter(log_folder + log_file_name +'_'+"inference.mp4",fourcc,15,(wide*2,height*1))

    n = input_thermal_frames.shape[0]
    for i in tqdm(range(n)):
        frame = input_thermal_frames[i, 0, :,:]
        # apply colormap on thermal image
        frame_norm = (frame - np.min(frame))/ (np.max(frame) - np.min(frame)) * 255
        frame_colormap = cv2.applyColorMap((frame_norm).astype(np.uint8), cv2.COLORMAP_JET) 
        groundtruth = groundtruth_3d_joints[i]
        prediction = estimated_3d_joints[i]
        vis_figure = draw_3D_landmarks2(frame,prediction,groundtruth)
        frame_colormap = cv2.resize(frame_colormap,t_size, interpolation=cv2.INTER_NEAREST)
        vis_figure = cv2.resize(vis_figure,t_size)
        concatenated_image = np.concatenate((frame_colormap,vis_figure),axis=1)
        if file_names_for_visualization is not None:
            cv2.putText(concatenated_image, file_names_for_visualization[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(concatenated_image)
    out.release()


def get_gpu_ids(i):
    gpu_ids = []
    gpu_info = os.popen("nvidia-smi -L").readlines()
    for line in gpu_info:
        # print(line)
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        gpu_ids.append(ids)
    if i >= len(gpu_ids):
        print("The number of the gpu is not enough! using the 0 by default")
        return gpu_ids[0]
    return gpu_ids[i]        

if __name__ == "__main__":
    # python edge_tapor_traintest.py -wp "77_20242421040_tapor.pth" -fs 1 -mt 1 -fd 336 -ld 64 -ft 0 -lr 0.00001 -bs 32 -ep 100 -dd 0 -v 0 
    parser = argparse.ArgumentParser(description='Knowledge Distiilation: Training and Testing')
    parser.add_argument('-wp', '--model_weight_file_name', type=str, default='.pth', help='the file name of the weights of the Tapor model')
    parser.add_argument('-mt', '--tapor_type', type=int, default=1, help='the type of tapor model: 0: small, 1: base1, 2: base2, 3: large1, 4: large2,5: base1_varaint1, 6: base1_variant2')
    parser.add_argument('-fs', '--fragment_size', type=int, default=1, help='the size of the fragment')
    
    parser.add_argument('-fd', '--feat_dim', type=int, default=336, help='the dim of the feature of the EdgeTapor model')
    parser.add_argument('-ld', '--last_layer_channel', type=int, default=64, help='the number of channels of the last layer of the EdgeTapor model')
    parser.add_argument('-ft', '--feature_type', type=int, default=0, help='0: no teacher, \
                                                    1: teacher cross-att output supervision, \
                                                    2: teacher self-att output supervision,')
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='the learning rate of the optimizer')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='the batch size of the training and testing')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='the epochs of the training')
    parser.add_argument('-sd', '--seed', type=int, default=0, help='the random seed')
    parser.add_argument('-dd', '--device', type=int, default=0, help='the device id')
    parser.add_argument('-v', '--video_flag', type=int, default=0, help='0: no visualization, 1: visualization')
    
    parser.add_argument('-ad', '--adaptor_flag', type=int, default=0, help='0: no adaptor, 1: adaptor')
    args = parser.parse_args()
    
    kd_train_dataset_save_path = "EdgeTapor_files/KD_training_data.pkl"
    
    model_weight_file_name = args.model_weight_file_name
    model_weight_path = "weights/" + model_weight_file_name # the path of the Tapor teacher model weights
    tapor_type = args.tapor_type
    fragment_size = args.fragment_size
    
    feat_dim = args.feat_dim
    last_layer_channel = args.last_layer_channel
    feature_type = args.feature_type
    
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    device = args.device
    video_flag = args.video_flag
    
    adaptor_flag = args.adaptor_flag

    # os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_ids(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # create folders to save the model weights and the training logs
    tensorboard_folder = 'EdgeTapor_files/edgeTapor_runs'
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    edge_weights_save_path = "EdgeTapor_files/edgeTapor_model_weights/"
    if not os.path.exists("EdgeTapor_files/edgeTapor_model_weights"):
        os.makedirs("EdgeTapor_files/edgeTapor_model_weights")
    log_folder = "EdgeTapor_files/edgeTapor_logs/"
    if not os.path.exists("EdgeTapor_files/edgeTapor_logs"):
        os.makedirs("EdgeTapor_files/edgeTapor_logs")
        
    localtime = time.localtime(time.time())
    index_of_experiment = len(os.listdir(tensorboard_folder))
    # the name for both the log file and the tensorboard log
    log_file_name = str(index_of_experiment) + "_" + str(localtime.tm_year) + str(localtime.tm_mon) + \
        str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec) + '_tm' + str(feature_type)
    if adaptor_flag == 1:
        log_file_name += '_adaptor'
    print("The log file name is: ",log_file_name)
    
    ## Loading the Tapor Model ##
    if tapor_type == 0:
        config = small_setting
    elif tapor_type == 1:
        config = base1_setting
    elif tapor_type == 2:
        config = base2_setting
    elif tapor_type == 3:
        config = large1_setting
    elif tapor_type == 4:
        config = large2_setting
    elif tapor_type == 5:
        config = base1_setting_varaint1
    elif tapor_type == 6:
        config = base1_setting_varaint2
    else:
        raise ValueError('The tapor type is not supported!')
    
    spatial_encoder_param = config['spatial_encoder_param']
    keypoints_encoder_param = config['keypoints_encoder_param']
    cross_keypoints_fusion_param = config['cross_keypoints_fusion_param']
    temporal_keypoints_fusion_param = config['temporal_keypoints_fusion_param']
    handpose_encoder_param = config['handpose_encoder_param']
    temporal_keypoints_fusion_param['num_history'] = fragment_size-1

    tapor_model = TaporTeacher(spatial_encoder_param, 
                keypoints_encoder_param, 
                cross_keypoints_fusion_param, 
                temporal_keypoints_fusion_param,
                handpose_encoder_param,
                input_width=32, 
                input_height=24,
                batch_size = batch_size,
                train=True,
                device=device).to(device)
    tapor_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    tapor_model.eval()
    
    edgetapor = EdgeTaporStudent(feat_dim,last_layer_channel).to(device)
    
    if adaptor_flag == 1:
        adaptor = Adaptor(21*48,feat_dim).to(device)
        dataset = KD_edgeTapor_dataset(kd_train_dataset_save_path, feature_type , feature_dim = 21*48)
    else:
        dataset = KD_edgeTapor_dataset(kd_train_dataset_save_path, feature_type, feat_dim)
    print("The size of the training dataset is: ",len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                [train_size, val_size]
                                                                )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    ## optimizer and loss function ##
    optimizer = torch.optim.Adam(edgetapor.parameters(), lr=learning_rate)
    if adaptor_flag == 1:
        optimizer = torch.optim.Adam(list(edgetapor.parameters()) + list(adaptor.parameters()), lr=learning_rate)
    # using two mse loss for the pose and the feature
    criterion_feat = nn.MSELoss()
    criterion_pose_mse = nn.MSELoss()
    criterion_bone_length = BoneLengthLoss()
    
    print("start training...")
    ## training ##
    writer = SummaryWriter(tensorboard_folder + '/' + log_file_name)
    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = None
    for epoch in range(epochs):
        edgetapor.train()
        if adaptor_flag == 1:
            adaptor.train()
        running_loss = 0.0
        running_loss_3d = 0.0
        running_loss_feat = 0.0
        running_loss_bone_length = 0.0
        for i, data in enumerate(train_loader):
            thermal_map, label, feature = data
            thermal_map = thermal_map.float().to(device)  
            label = label.float().to(device) 
            feature = feature.float().to(device) 
            if adaptor_flag == 1:
                print(feature.shape)
                feature = adaptor(feature)
            optimizer.zero_grad()
            outputs, feat = edgetapor(thermal_map)
            # reshape the last dim of outputs to 21*3
            outputs = outputs.view(-1, 21, 3)
            if feature_type == 1 or feature_type == 2:
                loss_3d = criterion_pose_mse(outputs, label)
                loss_bone_length = criterion_bone_length(outputs, label)
                loss_feat = criterion_feat(feat, feature)
                loss = loss_3d + 0.1*loss_feat + loss_bone_length
                running_loss_3d += loss_3d.item()
                running_loss_feat += 0.1*loss_feat.item()
                running_loss_bone_length += loss_bone_length.item()
            else:
                loss_3d = criterion_pose_mse(outputs, label)
                loss_bone_length = criterion_bone_length(outputs, label)
                loss = loss_3d + loss_bone_length
                running_loss_3d += loss_3d.item()
                running_loss_bone_length += loss_bone_length.item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0
                writer.add_scalar('training loss 3d', running_loss_3d / 10, epoch * len(train_loader) + i)
                running_loss_3d = 0.0
                writer.add_scalar('training loss feat', running_loss_feat / 10, epoch * len(train_loader) + i)
                running_loss_feat = 0.0
                writer.add_scalar('training loss bone length', running_loss_bone_length / 10, epoch * len(train_loader) + i)
                running_loss_bone_length = 0.0
                
        ## validation ##
        edgetapor.eval()
        if adaptor_flag == 1:
            adaptor.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                thermal_map, label, feature = data
                thermal_map = thermal_map.float().to(device)  
                label = label.float().to(device) 
                feature = feature.float().to(device) 
                if adaptor_flag == 1:
                    feature = adaptor(feature)
                outputs, feat = edgetapor(thermal_map)
                # reshape the last dim of outputs to 21*3
                outputs = outputs.view(-1, 21, 3)
                if feature_type == 1 or feature_type == 2:
                    loss_3d = criterion_pose_mse(outputs, label)
                    loss_bone_length = criterion_bone_length(outputs, label)
                    loss_feat = criterion_feat(feat, feature)
                    loss = loss_3d + 0.1*loss_feat + loss_bone_length
                    running_loss_3d += loss_3d.item()
                    running_loss_feat += 0.1*loss_feat.item()
                    running_loss_bone_length += loss_bone_length.item()
                else:
                    loss_3d = criterion_pose_mse(outputs, label)
                    loss_bone_length = criterion_bone_length(outputs, label)
                    loss = loss_3d + loss_bone_length
                    running_loss_3d += loss_3d.item()
                    running_loss_bone_length += loss_bone_length.item()
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader)
        print('Epoch %d, val loss: %.3f' % (epoch + 1, val_loss))
        writer.add_scalar('val loss', val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = edgetapor.state_dict()
    print('Finished Training')
    torch.save(best_weights, edge_weights_save_path + log_file_name + ".pth")    
    print('The best val loss is: ', best_val_loss)
    print('The best epoch is: ', best_epoch)        

    data_path = "Testset/"
    file_names = [
        'P1_U10_L_X_X_0.pkl',
        'P1_U10_L_X_X_1.pkl',
        'P1_U10_R_X_X_1.pkl',
        'P1_U11_R_X_X_0.pkl',
        'P1_U11_R_X_X_1.pkl',
        'P1_U1_L_X_X_1.pkl',
        'P1_U1_L_X_X_2.pkl',
        'P1_U1_R_Cover_coolwater_0.pkl',
        'P1_U1_R_Cover_coolwater_1.pkl',
        'P1_U1_R_Cover_flour_0.pkl',
        'P1_U1_R_Cover_flour_1.pkl',
        'P1_U1_R_Cover_foam_0.pkl',
        'P1_U1_R_Cover_foam_1.pkl',
        'P1_U1_R_Cover_no_0.pkl',
        'P1_U1_R_Cover_no_1.pkl',
        'P1_U1_R_Cover_oil_0.pkl',
        'P1_U1_R_Cover_oil_1.pkl',
        'P1_U1_R_Cover_warmwater_0.pkl',
        'P1_U1_R_Cover_warmwater_1.pkl',
        'P1_U1_L_thick1_0.pkl',
        'P1_U1_L_thick1_1.pkl',
        'P1_U1_L_thick1_10.pkl',
        'P1_U1_L_thick1_2.pkl',
        'P1_U1_L_thick1_3.pkl',
        'P1_U1_L_thick1_4.pkl',
        'P1_U1_L_thick1_5.pkl',
        'P1_U1_L_thick1_6.pkl',
        'P1_U1_L_thick1_7.pkl',
        'P1_U1_L_thick1_8.pkl',
        'P1_U1_L_thick1_9.pkl',
        'P1_U1_R_thick0_1.pkl',
        'P1_U1_R_thick0_10.pkl',
        'P1_U1_R_thick0_2.pkl',
        'P1_U1_R_thick0_7.pkl',
        'P1_U1_R_thick0_9.pkl',
        'P1_U1_R_thick1_0.pkl',
        'P1_U1_R_thick1_1.pkl',
        'P1_U1_R_thick1_10.pkl',
        'P1_U1_R_thick1_2.pkl',
        'P1_U1_R_thick1_3.pkl',
        'P1_U1_R_thick1_5.pkl',
        'P1_U1_R_thick1_7.pkl',
        'P1_U1_R_thick1_8.pkl',
        'P1_U1_R_thick1_9.pkl',
        'P1_U1_cover_medicalGlove_0.pkl',
        'P1_U1_cover_medicalGlove_1.pkl',
        'P1_U1_cover_medicalGlove_10.pkl',
        'P1_U1_cover_medicalGlove_2.pkl',
        'P1_U1_cover_medicalGlove_3.pkl',
        'P1_U1_cover_medicalGlove_4.pkl',
        'P1_U1_cover_medicalGlove_6.pkl',
        'P1_U1_cover_medicalGlove_7.pkl',
        'P1_U1_cover_medicalGlove_8.pkl',
        'P1_U1_cover_medicalGlove_9.pkl',
        'P1_UX_cover_nail_0.pkl',
        'P1_UX_cover_nail_2.pkl',
        'P1_UX_cover_nail_3.pkl',
        'P1_UX_cover_nail_4.pkl',
        'P1_UX_cover_nail_6.pkl',
        'P1_UX_cover_nail_7.pkl',
        'P1_U1_R_Dist_Dist_0.pkl',
        'P1_U1_R_Dist_Dist_1.pkl',
        'P1_U1_R_LightCond0_X_0.pkl',
        'P1_U1_R_LightCond0_X_1.pkl',
        'P1_U1_R_LightCond1_X_0.pkl',
        'P1_U1_R_LightCond1_X_1.pkl',
        'P1_U1_R_LightCond2_X_0.pkl',
        'P1_U1_R_LightCond2_X_1.pkl',
        'P1_U1_R_LightCond3_X_0.pkl',
        'P1_U1_R_LightCond3_X_1.pkl',
        'P1_U1_R_Cover_ring_0.pkl',
        'P1_U1_R_Cover_ring_1.pkl',
        'P1_U1_R_X_X_1.pkl',
        'P1_U2_R_Cover_coolwater_0.pkl',
        'P1_U2_R_Cover_coolwater_1.pkl',
        'P1_U2_R_Cover_flour_0.pkl',
        'P1_U2_R_Cover_flour_1.pkl',
        'P1_U2_R_Cover_foam_0.pkl',
        'P1_U2_R_Cover_foam_1.pkl',
        'P1_U2_R_Cover_no_0.pkl',
        'P1_U2_R_Cover_no_1.pkl',
        'P1_U2_R_Cover_oil_0.pkl',
        'P1_U2_R_Cover_oil_1.pkl',
        'P1_U2_R_Cover_warmwater_0.pkl',
        'P1_U2_R_Cover_warmwater_1.pkl',
        'P1_U2_R_Dist_Dist_0.pkl',
        'P1_U2_R_Dist_Dist_1.pkl',
        'P1_U2_R_Dist_Dist_2.pkl',
        'P1_U2_R_LightCond0_X_0.pkl',
        'P1_U2_R_LightCond0_X_1.pkl',
        'P1_U2_R_LightCond1_X_0.pkl',
        'P1_U2_R_LightCond1_X_1.pkl',
        'P1_U2_R_LightCond2_X_0.pkl',
        'P1_U2_R_LightCond2_X_1.pkl',
        'P1_U2_R_LightCond3_X_0.pkl',
        'P1_U2_R_LightCond3_X_1.pkl',
        'P1_U2_R_Cover_glove_1.pkl',
        'P1_U2_R_X_X_0.pkl',
        'P1_U2_R_X_X_1.pkl',
        'P1_U2_R_X_X_2.pkl',
        'P1_U3_L_X_X_1.pkl',
        'P1_U3_R_X_X_0.pkl',
        'P1_U4_L_X_X_0.pkl',
        'P1_U4_R_X_X_0.pkl',
        'P1_U5_R_X_X_0.pkl',
        'P1_U5_R_X_X_1.pkl',
        'P1_U6_L_X_X_0.pkl',
        'P1_U6_L_X_X_1.pkl',
        'P1_U6_R_X_X_0.pkl',
        'P1_U6_R_X_X_1.pkl',
        'P1_U7_L_X_X_1.pkl',
        'P1_U8_L_X_X_0.pkl',
        'P1_U8_R_X_X_0.pkl',
        'P1_U8_R_X_X_1.pkl',
        'P1_U9_R_X_X_0.pkl',
        'P2_U1_L_X_X_0.pkl',
        'P2_U1_L_X_X_2.pkl',
        'P2_U1_R_X_X_0.pkl',
        'P2_U2_L_X_X_0.pkl',
        'P2_U2_L_X_X_1.pkl',
        'P2_U2_L_X_X_2.pkl',
        'P2_U2_R_X_X_1.pkl',
        'P2_U2_R_X_X_2.pkl',
        'P3_U1_L_X_X_1.pkl',
        'P3_U1_L_X_X_2.pkl',
        'P3_U1_R_X_X_0.pkl',
        'P3_U1_R_X_X_1.pkl',
        'P3_U1_R_X_X_2.pkl',
        'P5_U1_L_X_X_0.pkl',
        'P5_U1_L_X_X_1.pkl',
        'P5_U1_L_X_X_2.pkl',
        'P5_U1_R_X_X_0.pkl',
        'P5_U1_R_X_X_1.pkl',
        'P5_U2_L_X_X_0.pkl',
        'P5_U2_L_X_X_1.pkl',
        'P5_U2_L_X_X_2.pkl',
        'P5_U2_R_X_X_0.pkl',
        'P5_U2_R_X_X_1.pkl',
        'P5_U2_R_X_X_2.pkl',
        'P6_U1_R_X_X_2.pkl',
        'P6_U1_L_X_X_1.pkl',
        'P6_U1_L_X_X_0.pkl',
        'P6_U1_R_X_X_1.pkl',
        ]

    edgetapor.eval()
    # using the best weights
    edgetapor.load_state_dict(best_weights)
    
    # processing all the files one by one
    result_dict = {
        'infernece_file_name': [],
        'failed_file_names': [],
        'predicts': [],
        'labels': [],
    }
    all_predicts = []
    all_labels = []
    all_input_thermal_frame = []
    failed_file_names = []
    file_names_for_visualization = []
    for file_name in tqdm(test_file_names):
        try:
            file_name_now = [file_name]
            test_dataset = iHand_dataset(test_data_path, file_name_now,fragment_size,map_size = (0,0), filter_out_no_hand=True, sliding_window = True)
            print("The size of the dataset is: ",len(dataset))
            dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last= True)
            predicts_dict,labels_dict = inference(edgetapor,dataloader,device)
            input_thermal_frame = np.concatenate(labels_dict['thermal_map'])
            predicts = np.concatenate(predicts_dict['joints_3d'])
            labels = np.concatenate(labels_dict['joints_3d'])          
            all_predicts.append(predicts)
            all_labels.append(labels)  
            all_input_thermal_frame.append(input_thermal_frame)    
            file_names_for_visualization += [file_name] * len(input_thermal_frame)    
            result_dict['infernece_file_name'].append(file_name)
            result_dict['predicts'].append(predicts)
            result_dict['labels'].append(labels)
            # calculating the mse and mae between the predicted landmarks and the ground truth landmarks
            print("The shape of the predicts and labels are: ")
            print(predicts.shape)
            print(labels.shape)
            average_mse_loss = np.mean((predicts - labels) ** 2)
            average_mae_loss = np.mean(np.abs(predicts - labels))
            per_dimention_mae_loss = np.mean(np.abs(predicts - labels),axis=0)
            per_dimention_mae_loss = np.mean(per_dimention_mae_loss,axis=0)
            per_dimention_mae_loss = per_dimention_mae_loss.tolist()
            mean_joints_error, mean_root_drift_error = calculate_shift_error(predicts, labels)
            mean_joints_error = mean_joints_error.tolist()
            mean_joints_error_str = [str(i) for i in mean_joints_error]
            mean_joints_error_str = ','.join(mean_joints_error_str)
            mpjpe = calculate_mpjpe(predicts, labels)
            pck_5mm = calculate_pck(predicts, labels, threshold=0.005)
            pck_1cm = calculate_pck(predicts, labels, threshold=0.01)
            pck_2cm = calculate_pck(predicts, labels, threshold=0.02)
            pck_3cm = calculate_pck(predicts, labels, threshold=0.03)
            pck_4cm = calculate_pck(predicts, labels, threshold=0.04)
            pck_5cm = calculate_pck(predicts, labels, threshold=0.05)
            pck_6cm = calculate_pck(predicts, labels, threshold=0.06)
            pck_7cm = calculate_pck(predicts, labels, threshold=0.07)
            pck_8cm = calculate_pck(predicts, labels, threshold=0.08)
            pck_9cm = calculate_pck(predicts, labels, threshold=0.09)
            print(f'Average MSE Loss(testset) {average_mse_loss:.10f}')
            print(f'Average MAE Loss(testset) {average_mae_loss:.10f}')
            print(f'Per Dimention MAE Loss(testset) {per_dimention_mae_loss}')
            print('Mean joints error:' + mean_joints_error_str + '\n')
            print(f'Mean root drift error: {mean_root_drift_error:.10f}')
            print(f'MPJPE {mpjpe:.10f}')
            print(f'PCK@5mm {pck_5mm:.10f}')
            print(f'PCK@1cm {pck_1cm:.10f}')
            print(f'PCK@2cm {pck_2cm:.10f}')
            print(f'PCK@3cm {pck_3cm:.10f}')
            print(f'PCK@4cm {pck_4cm:.10f}')
            print(f'PCK@5cm {pck_5cm:.10f}')
            print(f'PCK@6cm {pck_6cm:.10f}')
            print(f'PCK@7cm {pck_7cm:.10f}')
            print(f'PCK@8cm {pck_8cm:.10f}')
            print(f'PCK@9cm {pck_9cm:.10f}')
        except:
            failed_file_names.append(file_name)
            result_dict['failed_file_names'].append(file_name)
            print("The file: ",file_name," is not processed correctly!")
            continue

    print("The results are saved in: ",log_folder + log_file_name + '_log.txt')
    pickle.dump(result_dict,open(log_folder + log_file_name + '_result_dict.pkl','wb'))
    # create a video to visualize the results
    all_predicts = np.concatenate(all_predicts)
    all_labels = np.concatenate(all_labels)
    all_input_thermal_frame = np.concatenate(all_input_thermal_frame)
    if video_flag == 1:
        print("Generating the visualization video...")
        visualization_results_video(log_folder, log_file_name,all_input_thermal_frame, all_labels, all_predicts,(256*3,256*3), file_names_for_visualization)
        print("The visualization video is saved in: ",log_folder + log_file_name +'_'+"inference.mp4")
    else:
        print("The video is not generate!")
    print("The training and testing is finished!")