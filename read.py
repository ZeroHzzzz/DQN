import cv2
import numpy as np
class track:
    def __init__(self, path):
        self.path = path
        
    def __preprocess(self):
        self.track = cv2.imread(self.path)
        
        self.track = cv2.cvtColor(self.track, cv2.COLOR_BGR2GRAY)
        self.track = 255 - self.track
        
        gaussian_kernel = cv2.getGaussianKernel(9, 1.3)
        self.fus_track = cv2.filter2D(self.track, -1, gaussian_kernel)
        
        guss_noise = np.uint8(np.round(np.random.normal(15, 5, (300, 300))))
        guss_noise[guss_noise > 30] = 30
        max_val = np.max(self.fus_track)
        min_val = np.min(self.fus_track)
        self.fus_track = (self.fus_track - min_val) / (max_val - min_val)
        self.fus_track = np.clip(self.fus_track, 0, 225).astype(np.uint8)
        self.fus_track = self.fus_track + guss_noise
        return
    
    # def show_track(self):
    # def show_fus_track(self):
    
class camera_visio:
    def __init__(self, track):
        self.track = track
    def visio_ker(self, theta):
        X = np.mat([0.0], [0.0], [1.0])
        self.__update_trans_mat(theta)
        
        self.ker = np.ones((2*self.R+1, 2*self.R+1), dtype=np.uint8)
        for i in range(2*self.R+1):
            X[0,0] = i
            for j in range(2*self.R+1):
                X[1,0] = j
                self.ker[i,j] = self.__check(X)
        return self.ker
    
    def __check(self, X):
        X = X - np.mat([[self.R], [self.R], [0]])
        X = self.A * X
        temp = (X[0,0] - self.R) * self.T
        L3 = self.R + temp
        L4 = self.R - temp
        
        if(X[1,0] > self.L1 and X[1,0] < self.L2 and X[1,0] > L3):
            return 1
        else:
            return 0
    
    def visio_correction(self):
        rotated_image = cv2.warpAffine(self.vis, self.A_inv, (2*self.R+1))
        
        self.ct_vis = rotated_image[self.slice["y1"]:self.slice["y2"],
                                    self.slice["x1"]:self.slice["x2"]]
        return
    
    def visio_extraction(self, state):
        self.state = state
        self.visio_ker(self.state["theta"])
        
        xm = (self.state["X"][0,0] - self.R).astype(int)
        xM = (self.state["X"][0,0] + self.R).astype(int)
        ym = (self.state["X"][1,0] - self.R).astype(int)
        yM = (self.state["X"][1,0] + self.R).astype(int)
        self.vis = np.zeros((2*self.R+1, 2*self.R+1), dtype=float)
        
        for i in range(xm, xM+1):
            if  i < 1 or i > 300:
                continue
            for j in range(ym, yM+1):
                if j < 1 or j > 300:
                    continue
                self.vis[i - xm, j - ym] = self.track[i,j]* self.ker[i,j]
        return
    
    # def show_ker(self, theta):
    # def show_vis(self):
    # def show_ct_vis(self):
    
    def __preprocess(self):
        self.L1 = self.R + self.r * np.cos(np.pi/2)
        self.L2 = self.R + self.R * np.cos(np.pi/2)
        self.T = np.tan(np.pi/2 - self.Hphi)
        S = self.R * np.sin(np.pi/2 - self.Hphi)
        self.slice = {"x1":(self.R-S).astype(int),
                      "x2":(self.R+S).astype(int),
                      "y1":self.L1.astype(int),
                      "y2":self.L2.astype(int)
                      }
        return
    
    # def _update_trans_mat(self, theta):
    
# class AI:
#     def __init__(self, path):
#     def run(self, endT):
#     def __agent(self):
#     def __visio(self):
#     def __car_physical(self, agent):
#     def __save_to_txt(self):
        
# path = ""
# ai = AI(path)
# ai.run(30)