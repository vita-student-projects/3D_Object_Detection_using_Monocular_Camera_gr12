import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import json

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import GupnetLoss,Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.decode_helper import euler_to_quaternion

class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.curriculum = self.cfg_train.get('curriculum', None)   # treshold epochs
        print(f"Curriculum thresholds : {self.curriculum}")
    
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        
        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)


    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)
            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)                     
            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
            
            if (self.epoch % self.cfg_train['eval_frequency']) == 0:
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.eval_one_epoch()

            # save trained model
            if (self.epoch % self.cfg_train['save_frequency']) == 0:
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                
                # Curriculum learning
                if self.curriculum is not None:
                    targets = self.curriculum_learning(targets)

                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                criterion = GupnetLoss(self.epoch)
                outputs = self.model(inputs,coord_ranges,calibs,targets)
                _, loss_terms = criterion(outputs, targets)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict        
    
    def train_one_epoch(self,loss_weights=None):
        self.model.train()
        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            
            # Curriculum learning
            if self.curriculum is not None:
                targets = self.curriculum_learning(targets)

            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = GupnetLoss(self.epoch)
            outputs = self.model(inputs,coord_ranges,calibs,targets)
            total_loss, loss_terms = criterion(outputs, targets)
            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            if self.epoch == self.cfg_train['max_epoch']-1:
                total_loss.backward(retain_graph=False)
            else:
                total_loss.backward()
            self.optimizer.step()
            
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key] 
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]   
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
                            
        return stat_dict    
    
    def eval_one_epoch(self):
        self.model.eval()
        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device) 
                coord_ranges = coord_ranges.to(self.device)
    
                # the outputs of centernet
                outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val')

                dets = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()
                
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        cls_mean_size=cls_mean_size,
                                        thresholds = self.cfg_test['thresholds'])                 
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
            
        # save the result for evaluation.
        self.save_results_txt(results, self.cfg['output_dir'])
        # make a JSON for network of project 10
        self.save_results_JSON_group40(results, self.cfg['output_dir'])
        # make a JSON for DLAV lecture
        self.save_results_JSON_DLAV(results, self.cfg['output_dir'])
           
    def curriculum_learning(self, targets):

        mask_2d = targets.get('mask_2d')            # mask_2d (batch array)
        obj_num = targets.get('obj_num')            # number of objects (batch array)
        level = targets.get('obj_level')            # easy (1), moderate (2), hard(3)
        
        print_category = True

        for sample_nb in range(mask_2d.size()[0]):  # batch sample
            for i in range(obj_num[sample_nb]):     # objects in sample

                if 0 <= self.epoch <= self.curriculum[0] and level[sample_nb, i] != 1:
                    if print_category:
                        print('category : easy')
                        print_category = False
                    mask_2d[sample_nb, i] = False

                elif self.curriculum[0] < self.epoch <= self.curriculum[1] and level[sample_nb, i] != 2:
                    if print_category:
                        print('category : moderate')
                        print_category = False
                    mask_2d[sample_nb, i] = False

                elif self.curriculum[1] < self.epoch <= self.curriculum[2] and level[sample_nb, i] != 3:
                    if print_category:
                        print('category : hard')
                        print_category = False
                    mask_2d[sample_nb, i] = False

                elif self.curriculum[2] < self.epoch:
                    if print_category:
                        print('category : all')
                        print_category = False

        return targets
                
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
        
        # make a copy to JSON for network (XXX)
        self.save_results_JSON(results, output_dir)

    def save_results_JSON(self, results, output_dir='./outputs'):
        output_dir_json = os.path.join(output_dir, 'data')
        print(f"Saving JSON outputs to {output_dir_json}", flush=True)
        os.makedirs(output_dir_json, exist_ok=True)

        out_path = os.path.join(output_dir_json, 'inference.json')

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




