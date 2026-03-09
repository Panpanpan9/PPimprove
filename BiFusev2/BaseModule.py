# BaseModule.py
# 作用：作为模型的基类，封装模型保存/加载逻辑（checkpoint 管理）

import os
import torch
import torch.nn as nn
import datetime

class BaseModule(nn.Module):
    def __init__(self, path):
        super().__init__()
        # 保存目录路径
        self.path = path
        # 创建目录，若存在则无需操作
        os.system('mkdir -p %s'%path)

        # 列出目录下所有以.pkl 结尾的文件 并排序，作为已存在 checkpoint 列表
        # 此处的 model_list 会用于 Load()中选择要加载的模型文件
        self.model_lst = [x for x in sorted(os.listdir(self.path)) if x.endswith('.pkl')]
        
        # 用于记录当前 最优模型 的文件名
        self.best_model = None
        # 存储最优模型对应的 accuracy（用于判断是否替换）
        self.best_accuracy = -float('inf')
    
    def _loadName(self, epoch=None):
        """
        内部方法：根据给定 epoch 或默认策略决定要加载的 checkpoint 文件名
        返回： (name, index)
        - name: 要加载的文件名（如果没有合适文件，返回 None）
        - index: name 在 self.model_lst 中的索引（若 name 为 None，返回 0）
        行为说明：
        - 如果目录为空，打印提示并返回 (None, 0)
        - 如果传入 epoch（整数），尝试在文件列表中查找 file 名以 'xxxxx.pkl' 末尾包含该 epoch 的文件
          如果找到则返回对应文件名；否则打印找不到并返回 (None, 0)
        - 如果 epoch 为 None，则返回列表中最后一个文件（字符串排序后的末项）
        """
        # 若保存目录为空，则没有可加载的模型
        if len(self.model_lst) == 0:
            print("Empty model folder! Using initial weights")
            return None, 0

        # 若用户指定了 epoch，则尝试找到对应 epoch 的文件（文件命名约定在 Save 中给出）
        if epoch is not None:
            # model_list: 所有以.pkl 结尾的文件列表
            for i, name in enumerate(self.model_lst):
                 # 文件名末尾应以 epoch 的五位格式结尾，例如 '00012.pkl'
                if name.endswith('%.5d.pkl'%epoch):
                    print("Use %s"%name)
                    return name, i
            # 若未找到对应 epoch 文件，告知用户并返回 None（使用初始权重）
            print ('Epoch not found, use initial weights')
            return None, 0
        else:
            # 如果未指定 epoch，则默认使用文件列表中的最后一个（最近的一个，取决于文件名排序）
            print ('Use last epoch, %s'%self.model_lst[-1])
            return self.model_lst[-1], len(self.model_lst)-1

    def Load(self, epoch=None):
        name, _ = self._loadName(epoch)
        if name is not None:
            params = torch.load('%s/%s'%(self.path, name), map_location='cpu')
            self.load_state_dict(params, strict=False)
            self.best_model = name
            epoch = int(self.best_model.split('_')[-1].split('.')[0]) + 1
        else:
            epoch = 0

        return epoch

    def Save(self, epoch, accuracy=None, replace=False):
        if accuracy is None or replace==False:
            aaa = '%.5d'%epoch
            now = 'model_%s.pkl'%datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
            params = self.state_dict()
            name = '%s/%s'%(self.path, now)
            torch.save(params, name)
            self.best_model = now
        else:
            if accuracy > self.best_accuracy:
                aaa = '%.5d'%epoch
                now = 'model_%s.pkl'%datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_{}'.format(aaa))
                params = self.state_dict()
                name = '%s/%s'%(self.path, now)
                if self.best_model is not None: os.system('rm %s/%s'%(self.path, self.best_model))
                torch.save(params, name)
                self.best_model = now
                self.best_accuracy = accuracy
                print ('Save %s'%name)
