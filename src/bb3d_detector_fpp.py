import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
import torch
from torch.utils.data import DataLoader
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network, example_convert_to_torch
from second.protos import pipeline_pb2
from second.utils import config_tool
from second.pytorch.builder.input_reader_builder import DatasetWrapper
from second.data.preprocess import merge_second_batch
from second.data.inference_dataset import InferDataset

class FrustumPP_Detector:

    def __init__(self, model_path):
        self.saved_model_path = model_path
        config_path = self.saved_model_path+"pipeline.config"
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        self.model_cfg = self.config.model.second
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        self.train_cfg = self.config.train_config
        ckpt_path = self.saved_model_path+"voxelnet-321600.tckpt"
        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(ckpt_path))
        if self.train_cfg.enable_mixed_precision:
            self.net.half()
            self.net.metrics_to_float()
            self.net.convert_norm_to_float(self.net)
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

    def generate_anchors(self):
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]
        self.anchors = self.target_assigner.generate_anchors(feature_map_size)["anchors"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)

    def load_pointclouds(self, pcls, objs_type):

        def load_pointcloud(points, obj_type):
            points = points.T
            dict_voxel = self.voxel_generator.generate(points, max_voxels=500)

            coords = np.pad(dict_voxel['coordinates'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
            coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
            voxels = torch.tensor(dict_voxel['voxels'], dtype=torch.float32, device=self.device)
            num_voxels = torch.tensor(dict_voxel['voxel_num'], dtype=torch.float32, device=self.device)
            num_points = torch.tensor(dict_voxel['num_points_per_voxel'], dtype=torch.int32, device=self.device)

            ff_input = {
                "anchors": self.anchors.cpu(),
                "voxels": voxels.cpu(),
                "num_points": num_points.cpu(),
                "num_voxels": num_voxels.cpu(),
                "coordinates": coords.cpu(),
                "type": obj_type,
            }

            return ff_input
        
        ff_inputs = np.array(list(map(load_pointcloud, pcls, objs_type)))
        dataset = InferDataset(ff_inputs)
        self.eval_dataloader = DataLoader(dataset, batch_size=20, shuffle=False, pin_memory=False, collate_fn=merge_second_batch)
        self.float_dtype = None
        if self.train_cfg.enable_mixed_precision:
            self.float_dtype = torch.float16
        else:
            self.float_dtype = torch.float32

    def detect(self):
        fpp_detections = []
        for data in self.eval_dataloader:
            data = example_convert_to_torch(data, self.float_dtype)
            with torch.no_grad():
                fpp_detections += self.net(data)
                print(fpp_detections)
        return fpp_detections