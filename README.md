# demo_shaobin
# NIO_shaobin_2021-12-23
# 大范围移动，完整代码
# 包括处理，预测，滤波
import json
import os
from turtle import left

""" Vanilla implementation of the standard Kalman filter algorithm"""
import numpy as np
from typing import List, Tuple

from numpy.lib.function_base import diff
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import glob
from resnet_pytorch import ResNet 
import matplotlib.pyplot as plt
import get_recall as f_yym
import pdb
import time
# import draw_curve as y


class KalmanFilter:

    def __init__(
        self, A: np.ndarray,
        xk: np.ndarray,
        B: np.ndarray,
        Pk: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray
    ):
        self.A = A
        self.xk = xk
        self.B = B
        self.Pk = Pk
        self.H = H
        self.Q = Q
        self.R = R   

        # attributes
        self.state_size = self.xk.shape[0] # usually called 'n'
        self.__I = np.identity(self.state_size)
        self.kalman_gains = []

    def predict(
        self,
        Ak: np.ndarray,
        xk: np.ndarray,
        Bk: np.ndarray,
        uk: np.ndarray,
        Pk: np.ndarray,
        Qk: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # project state ahead
        xk_prior = Ak @ xk + Bk @ uk
        
        # project error covariance ahead
        Pk_prior = Ak @ ((Pk @ Ak.T) + Qk)
        
        return xk_prior, Pk_prior
    
    def update(
        self,
        Hk: np.ndarray,
        xk: np.ndarray,
        Pk: np.ndarray,
        zk: np.ndarray,
        Rk:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates states and covariances.

        Update step of the Kalman filter. That is, the filter combines the 
        predictions with the observed variable :math:`Z` at time :math:`k`.

        Parameters
        ----------
        Hk : numpy.ndarray
            Observation matrix at time :math:`k`.
        xk : numpy.ndarray
            Prior mean state estimate at time :math:`k`.
        Pk : numpy.ndarray
            Prior covariance state estimate at time :math:`k`.
        zk : numpy.ndarray
            Observation at time :math:`k`.
        Rk : numpy.ndarray
            Measurement noise covariance at time :math:`k`.
            
        Returns
        -------
        xk_posterior : numpy.ndarray
            A posteriori estimate error mean at time :math:`k`.
        Pk_posterior : numpy.ndarray
            A posteriori estimate error covariance at time :math:`k`.
        """
        # innovation (pre-fit residual) covariance
        Sk = Hk @ (Pk @ Hk.T) + Rk
        
        # optimal kalman gain
        Kk = Pk @ (Hk.T @ np.linalg.inv(Sk))
        self.kalman_gains.append(Kk)

        # update estimate via zk
        xk_posterior = xk + Kk @ (zk - Hk @ xk)
        
        # update error covariance
        Pk_posterior = (self.__I - Kk @ Hk) @ Pk
        
        return xk_posterior, Pk_posterior
    
    def filter(
        self,
        Z: np.ndarray,
        U: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run filter over Z and U.
        
        Applies the filtering process over :math:`Z` and :math:`U` and returns 
        all errors and covariances. That is: given :math:`Z` and :math:`U`, 
        this function applies the predict and update feedback loop for each 
        :math:`zk`, where :math:`k` is a timestamp.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Observed variable
        U : numpy.ndarray
            Control-input vector.
            
        Returns
        -------
        states : list of numpy.ndarray
            A posteriori state estimates for each time step :math:`k`.
        errors : list of numpy.ndarray
            A posteriori estimate error covariances for each time step 
            :math:`k`.
        """
        states = []
        errors = []
        
        # get initial conditions
        xk = self.xk
        Pk = self.Pk
        
        # feedback-control loop
        _iterable = zip(self.A, self.H, self.B, U, Z, self.Q, self.R)
        for k, (Ak, Hk, Bk, uk, zk, Qk, Rk) in enumerate(_iterable):
            # predict step, get prior estimates
            xk_prior, Pk_prior = self.predict(
                Ak=Ak,
                xk=xk,
                Bk=Bk,
                uk=uk,
                Pk=Pk,
                Qk=Qk
            )
            
            # update step, correct prior estimates
            xk_posterior, Pk_posterior = self.update(
                Hk=Hk,
                xk=xk_prior,
                Pk=Pk_prior,
                zk=zk,
                Rk=Rk
            )

            states.append(xk_posterior)
            errors.append(Pk_posterior)

            # update estimates for the next iteration
            xk = xk_posterior
            Pk = Pk_posterior
            
        return states, errors


class Large_Object_Move_Detection:
    def __init__(self):
        # 超参数
        self.diff_img_threshold = 0.05
        self.diff_num_threshold = 50
        self.q = 0.005
        self.r = 0.1
        
        # Preprocess image
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(346),
                #transforms.CenterCrop(1680),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        # result
        self.keep = []
        self.keep_first_img_index = []
        self.predict_data = []
        # data input
        self.select_scene = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.select_mask = ["left","left","front","front","left","right","left","front","front","left","front","left","right"]
        self.all_data_path = ["/media/nio/backup/yinyiming/project/data/test_image_20220112/n1/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n2/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n3/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n4/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n5/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n6/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/n7/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p1/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p2/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p3/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p4/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p5/*.png",
                    "/media/nio/backup/yinyiming/project/data/test_image_20220112/p6/*.png"]


    def input_path(self, a = None, path1_mask = None, path2_pic = None):
        # input:    输入，选择的场景，n1场景输入0，n2输入1
        # return:   输出, 对应场景的路径以及此视角的mask路径

        # 异常错误
        if a and a not in self.select_scene:
            print("a error!")
            
        elif a is None and not path1_mask:
            print("input is None!")

        elif a in self.select_scene:
            self.a = a
            if self.select_mask[self.a] == "front":
                roi_path = glob.glob("/media/nio/backup/yinyiming/project/mask/front_mask.png")
            elif self.select_mask[self.a] == "left":
                roi_path = glob.glob("/media/nio/backup/yinyiming/project/mask/left_mask.png")
            elif self.select_mask[self.a] == "right":
                roi_path = glob.glob("/media/nio/backup/yinyiming/project/mask/right_mask.png")
            elif self.select_mask[self.a] == "back":
                roi_path = glob.glob("/media/nio/backup/yinyiming/project/mask/back_mask.png")

            self.roi_path = roi_path[0]
            self.data_path = self.all_data_path[self.a]

        elif path1_mask and path2_pic:
            self.roi_path = path1_mask
            self.data_path = path2_pic


    def preprocessing(self,input,pic2):
        # 输入图片与mask路径
        # 输出处理过后的两张图片

        roi = Image.open(self.roi_path)
        roi = roi.resize((1920, 1536))
        roi = np.array(roi)
    
        # mask_fiename = data_path.split("/")[-2] + "_mask.png"
        # cv2.imwrite(mask_fiename, roi)

        zero = np.zeros((1536,1920,3))
        zero[:,:,0] = roi
        zero[:,:,1] = roi
        zero[:,:,2] = roi
        roi_train = zero
        roi_train = torch.from_numpy(roi_train)

        index = int(input.split("/")[-1].split("-")[1].split(".")[0])
        imgname1 = '/'.join(self.data_path.split("/")[0:-1]) + "/image-" + str(index).zfill(4) + ".png"
        imgname2 = '/'.join(self.data_path.split("/")[0:-1]) + "/image-" + str(pic2).zfill(4) + ".png"
        
        input_image = cv2.imread(imgname1)
        input_image2 = cv2.imread(imgname2)

        input_image = torch.from_numpy(input_image)
        input_image2 = torch.from_numpy(input_image2)


        roi_boundry = np.where(roi == 255)

        y_max = np.max(roi_boundry[0])
        x_max = np.max(roi_boundry[1])
        y_min = np.min(roi_boundry[0])
        x_min = np.min(roi_boundry[1])
        
        # (1536, 1920, 3)
        input_ori = input_image*roi_train
        input_ori = input_ori.numpy()
        # (650, 1919, 3)
        tmp = np.zeros((y_max - y_min, x_max - x_min, 3))
        tmp[:, :, :] = input_ori[y_min:y_max, x_min:x_max, :]
        # 重新拼成1300*1300
        Square_side_length = (y_max - y_min)*2
        ssl = Square_side_length

        input = np.zeros((ssl, ssl, 3))
        input[0:(y_max - y_min), :ssl, :] = tmp[:, :ssl, :]
        input[(y_max - y_min):, :(x_max - x_min - ssl)] = tmp[:, ssl:, :]


        input2_ori = input_image2*roi_train
        input2_ori = input2_ori.numpy()
        tmp = np.zeros((y_max - y_min, x_max - x_min, 3))
        tmp[:, :, :] = input2_ori[y_min:y_max, x_min:x_max, :]

        input2 = np.zeros((ssl, ssl, 3))
        input2[0:(y_max - y_min), :ssl, :] = tmp[:, :ssl, :]
        input2[(y_max - y_min):, :(x_max - x_min - ssl)] = tmp[:, ssl:, :]


        # float64的输入，to tensor不会转成[0,1]，因为bn层要求输入是【-1，1】
        # print(input.dtype)
        
        input = 255 * input # Now scale by 255
        input2 = 255 * input2
        # print(input.dtype)
        # pdb.set_trace()
        
        input = input.astype(np.uint8)
        input_tensor = self.preprocess(input) 
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        input2 = input2.astype(np.uint8)
        input_tensor2 = self.preprocess(input2)
        input_batch2 = input_tensor2.unsqueeze(0)
        print(input_batch2.dtype)

        return input_batch, input_batch2


    def test_kalman(self,data):
        # Parameters
        # ----------
        # A : numpy.ndarray
        #     Transition matrix. A matrix that relates the state at the previous time 
        #     step k-1 to the state at the current step :math:`k`.
        # xk : numpy.ndarray
        #     Initial (:math:`k=0`) mean estimate.
        # B : numpy.ndarray
        #     Control-input matrix.
        # Pk : numpy.ndarray
        #     Initial (:math:`k=0`) covariance estimate.
        # H : numpy.ndarray
        #     Observation matrix. A matrix that relates the state to the measurement 
        #     :math:`z_{k}`.
        # Q : numpy.ndarray
        #     Process noise covariance (transition covariance).
        # R : numpy.ndarray or float.
        #     Measurement noise covariance (observation covariance).

        # set parameters
        Z = data
        # !!!A0.5
        # 这个状态转移方程不能改！！！默认为1
        A = np.expand_dims(np.ones((len(Z),1)), axis=1)
        # mean estimate.
        xk = np.array([[0]])
        # Control-input matrix. 这个地方修改！！！zeros
        B = np.expand_dims(np.zeros((len(Z),1)), axis=1)

        U = np.zeros((len(Z), 1))
        # covariance estimate.  
        # 1改成0.5
        Pk = np.array([[1]])
        # Observation matrix. A matrix that relates the state to the measurement
        H = np.expand_dims(np.ones((len(Z),1)), axis=1)
        # Q是主要手段！！！
        #  Q = np.zeros((len(Z)))
        # R固定，Q越大，代表越信任侧量值，Q无穷代表只用测量值；反之，Q越小代表越信任模型预测值，Q为零则是只用模型预测。
        Q = np.ones((len(Z))) * self.q
        # 改这个貌似没用 哪个是alpha
        R = np.ones((len(Z))) * self.r

        kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
        states, covariances = kf.filter(Z=Z, U=U)

        return states, covariances


    def get_results(self):

        # 1. 打开所有图片，
        imglists = glob.glob(self.data_path)
        imglists.sort()

        # start = time.clock()
        for pic1_path in imglists:
            pic2 = int(pic1_path.split("/")[-1].split("-")[1].split(".")[0]) + 1
            if pic2 > int(imglists[-1].split("/")[-1].split("-")[1].split(".")[0]):
                continue
            index = int(pic1_path.split("/")[-1].split("-")[1].split(".")[0])

            # 2. 预处理图片
            input_batch, input_batch2 = self.preprocessing(pic1_path, pic2)

            # 3. 评估，不训练
            model = ResNet.from_pretrained("resnet18")
            model.eval()

            with torch.no_grad():
                logits = model.extract_features(input_batch.to(torch.float32))
                logits = logits.mean(dim=1)

                logits2 = model.extract_features(input_batch2.to(torch.float32))
                logits2 = logits2.mean(dim=1)

                diff_fe = abs(logits2 - logits)

                # print("diff max: ", diff_fe.max())
                # print("diff min: ", diff_fe.min())

                diff_img = diff_fe.numpy()[0, :, :]

                # 返回坐标(0,0)或（ ，）
                diff_index = np.where(diff_img > self.diff_img_threshold)
                diff_num = len(diff_index[0])
                print("12*15的图片上出现了{}个差异大于阈值的像素点".format(diff_num))

                # 可视化；x:每组第一张图片下标 y：每组的差异像素点个数
                self.keep.append(diff_num)
                self.keep_first_img_index.append(index)

            print("-----")
            # move the input and model to GPU for speed if available
            # if torch.cuda.is_available():
            #     input_batch = input_batch.to("cuda")
            #     input_batch2 = input_batch2.to("cuda")
            #     model.to("cuda")

        self.predict_data,cor = self.test_kalman(self.keep)

        # end = time.clock()
        # print(end-start)

        return self.predict_data, self.keep_first_img_index, self.keep

 
def draw_my_curve(x,y1,y2,filename):
    
    fig = plt.figure()
    plt.scatter(x, y1, c='red', marker='.')
    plt.scatter(x, y2, c='blue', marker='.')
    plt.savefig(filename)
    
LO = Large_Object_Move_Detection()
# LO.input_path(path1_mask="/media/nio/backup/yinyiming/project/mask/front_mask.png",path2_pic="/media/nio/backup/yinyiming/project/data/test_image_20220112/n3/*.png")

LO.input_path(a=0)
LO.get_results()

draw_my_curve(LO.keep_first_img_index, LO.predict_data, LO.keep, "/media/nio/backup/yinyiming/project/YYM.png")



