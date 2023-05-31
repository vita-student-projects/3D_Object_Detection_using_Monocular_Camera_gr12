import os
import tqdm

import torch
import numpy as np
import json

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.decode_helper import euler_to_quaternion

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            print(f"Using pre-trained model with weights located at {cfg['resume_model']}", flush=True)
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=50)
            dets = dets.detach().cpu().numpy()


            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size

            dets = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     thresholds = self.cfg['thresholds'])

            results.update(dets)
            progress_bar.update()

        # save the result for evaluation.
        self.save_results_txt(results, self.cfg['output_dir'])
        # make a JSON for network of project 10
        self.save_results_JSON_group40(results, self.cfg['output_dir'])
        # make a JSON for DLAV lecture
        self.save_results_JSON_DLAV(results, self.cfg['output_dir'])
        progress_bar.close()

    def save_results_txt(self, results, output_dir='./outputs'):
        output_dir_txt = os.path.join(output_dir, 'data')
        print(f"Saving TXT outputs to {output_dir_txt}", flush=True)
        os.makedirs(output_dir_txt, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir_txt, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def save_results_JSON_group40(self, results, output_dir='./outputs'):
        output_dir_json = os.path.join(output_dir, 'data')
        print(f"Saving JSON outputs to {output_dir_json}", flush=True)
        os.makedirs(output_dir_json, exist_ok=True)

        out_path = os.path.join(output_dir_json, 'inference_group40.json')

        meta_dict = {
            "use_camera": True, 
            "use_lidar": False, 
            "use_radar": False, 
            "use_map": False, 
            "use_external": False}
        
        result_dict = {}
        
        # for each image
        for img_id in results.keys():
            # for each object in the iamge

            objects_list = []

            for i in range(len(results[img_id])):

                o = results[img_id][i]
                object_dict = {
                    "sample_token": int(img_id),
                    "translation": [float(o[9]), float(o[10]), float(o[11])],
                    "size": [float(o[7]), float(o[8]), float(o[6])],
                    "rotation": euler_to_quaternion(0, 0, float(o[12])),
                    "velocity": [],
                    "detection_name": self.class_name[int(o[0])],
                    "detection_score": float(o[13]),
                    "attribute_name": "",
                    "kitti_ry": float(o[12])
                }
                objects_list.append(object_dict)

            result_dict.update({int(img_id):objects_list})

        json_string = {"results": result_dict, "meta": meta_dict}
        
        with open(out_path, 'w') as f:
            json.dump(json_string, f)

    def save_results_JSON_DLAV(self, results, output_dir='./outputs'):
        output_dir_json = os.path.join(output_dir, 'data')
        print(f"Saving JSON outputs to {output_dir_json}", flush=True)
        os.makedirs(output_dir_json, exist_ok=True)

        out_path = os.path.join(output_dir_json, 'inference_DLAV.json')

        meta_dict = {
            "use_camera": True, 
            "use_lidar": False, 
            "use_radar": False, 
            "use_map": False, 
            "use_external": False}
        
        frame_list = []
        
        # for each image
        for img_id in results.keys():
            # for each object in the iamge

            objects_list = []

            for i in range(len(results[img_id])):

                o = results[img_id][i]
                object_dict = {
                    "sample_token": int(img_id),
                    "translation": [float(o[9]), float(o[10]), float(o[11])],
                    "size": [float(o[7]), float(o[8]), float(o[6])],
                    "rotation": euler_to_quaternion(0, 0, float(o[12])),
                    "velocity": [],
                    "detection_name": self.class_name[int(o[0])],
                    "detection_score": float(o[13]),
                    "attribute_name": "",
                    "kitti_ry": float(o[12])
                }
                objects_list.append(object_dict)

            frame_list.append({"frame": int(img_id), "predictions":objects_list})

        json_string = {"project": "8. 3D Object Detection using Monocular Camera", "outputs": frame_list}
        
        with open(out_path, 'w') as f:
            json.dump(json_string, f)




