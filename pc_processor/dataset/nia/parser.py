import os
import yaml
import numpy as np
from PIL import Image
import cv2
import open3d as o3d

class NIADataset(object):
    def __init__(self, root,  # directory where data is
                 config_path,  # directory of config file
                 mode="train"):
        self.root = root
        self.mode = mode # train or val or test

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []
        self.label_files = []
        self.image_files = []
        #self.proj_matrix = {}
        self.sequences = self.data_config["sequences"][self.mode]

        for seq in self.sequences:
            print("parsing seq {}...".format(seq))
            
            # find label mask first
            label_path = os.path.join(self.root, seq, "seg/mask")
            label_files =  [os.path.join(label_path, f)
                            for f in os.listdir(label_path) if ".png" in f]
            label_ids = [val.split("/")[-1].split(".")[0] for val in label_files]
            
            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, "refine/pcd")
            pointcloud_files = [os.path.join(pointcloud_path, f) 
                                for f in os.listdir(pointcloud_path) if (".pcd" in f) and (f.split(".")[0] in label_ids)]
            pointcloud_ids = [val.split("/")[-1].split(".")[0] for val in pointcloud_files]

            image_path = os.path.join(self.root, seq, "refine/camera")
            image_files = [os.path.join(image_path, f) 
                           for f in os.listdir(image_path) if (".png" in f) and (f.split(".")[0] in label_ids)]
            image_ids = [val.split("/")[-1].split(".")[0] for val in image_files]
            
            # find unmatched files and remove them
            skip_idx = []
            for i in range(len(label_ids)):
                is_omitted = False
                if not label_ids[i] in pointcloud_ids:
                    is_omitted = True
                if not label_ids[i] in image_ids:
                    is_omitted = True
                if is_omitted:
                    skip_idx.append(i)

            for idx in skip_idx:
                val = label_ids[idx]
                if val in pointcloud_ids:
                    pcd_idx = pointcloud_ids.index(val)
                    del pointcloud_files[pcd_idx]
                if val in image_ids:
                    img_idx = image_ids.index(val)
                    del image_files[img_idx]
                del label_files[idx]
                  
            assert (len(label_files) == len(pointcloud_files))
            assert (len(label_files) == len(image_files))
            
            self.label_files.extend(label_files)
            self.pointcloud_files.extend(pointcloud_files)
            self.image_files.extend(image_files)

            # load calibration file
            calib_dir = os.path.join(self.root, seq.split("/")[0])
            calib = self.read_calib(calib_dir)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix = proj_matrix

        # sort for correspondance
        self.pointcloud_files.sort()
        self.label_files.sort()
        self.image_files.sort()
        print("Using {} pointclouds from sequences {}".format(
            len(self.pointcloud_files), self.sequences))

        # load config -------------------------------------
        # get color map
        sem_color_map = self.data_config["color_map"]
        num_color_map = len(sem_color_map)
        self.sem_color_lut = np.zeros((num_color_map, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0
        self.class_map_lut = np.arange(num_color_map, dtype=np.int32)
        self.class_map_lut_inv = np.arange(num_color_map, dtype=np.int32)
        self.mapped_cls_name = self.data_config["labels"]

    @staticmethod
    def read_calib(calib_dir):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        cam = np.zeros([3,4])
        with open(os.path.join(calib_dir, "cam.txt"), 'r') as f:
            idx = 0
            for line in f.readlines():
                if line == '\n':
                    break
                cam[idx,:3] = [float(val) for val in line.split(" ")]
                idx += 1
        calib = np.identity(4)
        with open(os.path.join(calib_dir, "calib.txt"), 'r') as f:
            idx = 0
            for line in f.readlines():
                if line == '\n':
                    break
                calib[idx,:] = [float(val) for val in line.split(" ")]
                idx += 1

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out['P2'] = cam
        calib_out['Tr'] = calib
        return calib_out

    @staticmethod
    def readPCD(path):
        pointcloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(str(path), "pcd")
        xyz = np.asarray(pointcloud.points).copy()
        del pointcloud
        xyz = xyz[:, :3].astype(np.float32)
        return xyz

    @staticmethod
    def readLabel(path):
        label = cv2.imread(path)
        return label

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if "\\" in path:
            # windows path
            path_split = path.split("\\")
            num_delim_root = len(self.root.split("\\"))
        else:
            path_split = path.split("/")
            num_delim_root = len(self.root.split("/"))
        seq_id = "/".join(path_split[num_delim_root:-1])
        frame_id = path_split[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByIndex(self, index):
        sem_mask = self.readLabel(self.label_files[index])
        sem_label = np.zeros(sem_mask.shape[:2])
        for label_id in range(len(self.sem_color_lut)):
            mask = np.all(sem_mask == self.sem_color_lut[label_id,:], axis=2)
            sem_label[mask] = label_id
        return sem_label, None

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        sem_mask = self.readLabel(self.label_files[index])
        sem_label = np.zeros(sem_mask.shape[:2])
        for label_id in range(len(self.sem_color_lut)):
            mask = np.all(sem_mask == self.sem_color_lut[label_id,:], axis=2)
            sem_label[mask] = label_id

        return pointcloud, sem_label, None

    def loadImage(self, index):
        return Image.open(self.image_files[index])

    def mapLidar2Camera(self, seq, pointcloud, img_h, img_w):
        proj_matrx = self.proj_matrix
        # only keep point in front of the vehicle
        keep_mask = pointcloud[:, 0] > 0
        pointcloud_hcoord = np.concatenate([pointcloud[keep_mask], np.ones(
            [keep_mask.sum(), 1], dtype=np.float32)], axis=1)
        mapped_points = (proj_matrx @ pointcloud_hcoord.T).T  # n, 3
        # scale 2D points
        mapped_points = mapped_points[:, :2] / \
                        np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
        keep_idx_pts = (mapped_points[:, 0] > 0) * (mapped_points[:, 0] < img_h) * (
                mapped_points[:, 1] > 0) * (mapped_points[:, 1] < img_w)
        keep_mask[keep_mask] = keep_idx_pts
        # fliplr so that indexing is row, col and not col, row
        mapped_points = np.fliplr(mapped_points)
        return mapped_points[keep_idx_pts], keep_mask

    def __len__(self):
        return len(self.pointcloud_files)