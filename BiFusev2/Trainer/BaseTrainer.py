import os
import abc
import numpy as np
import time
import socket
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from .. import Tools, Dataset, Metric
# import BiFusev2.Tools as Tools
# import BiFusev2.Dataset as Dataset
# import BiFusev2.Metric as Metric

def ScriptStart(args, config, litmodule):
    if args.mode == 'train':
        logger_type = config['exp_args'].get('logger_type', 'TensorBoardLogger')
        num_nodes = config['exp_args'].get('num_nodes', 1)
        check_val_every_n_epoch = config['exp_args'].get('check_val_every_n_epoch', 1)
        devices = config['exp_args'].get('devices', None)
        if logger_type == 'TensorBoardLogger':
            logger = pl_loggers.TensorBoardLogger(config['exp_args']['exp_path'])
        elif logger_type == 'WandbLogger':
            name = os.getcwd().split('/')[-1]
            assert name != 'pack'
            #name = 'GGG'
            name = '%s:%s'%(socket.gethostname(), name)
            os.system('mkdir %s'%config['exp_args']['exp_path'])
            logger = pl_loggers.WandbLogger(
                project='BiFuse++', 
                name=name, 
                save_dir=config['exp_args']['exp_path']
            )
        else:
            raise ValueError('Logger type weird')
        st = 'ddp'
        litmodule.SetStrategy(st)

        # 【20251224 新增 】使得 TenserBoard 根据是否加载模型 调整起止的epoch（开始）
        # 计算本次任务实际需要跑的轮数：总目标轮数 (Y) - 已经跑过的轮数 (offset)
        total_target_epochs = config['exp_args']['epoch']
        remaining_epochs = total_target_epochs - litmodule.epoch_offset
        
        if remaining_epochs <= 0:
            print(f"Training already reached target epoch {total_target_epochs}!")
            return

        print(f"Training Plan: Run for {remaining_epochs} epochs (Target: {total_target_epochs})")
        # 【20251224 新增 】 计算本次任务实际需要跑的轮数：使得 TenserBoard 根据是否加载模型 调整起止的epoch（结束）

        # '''
        # 为了适配 devices（4 -> 1）和 batchsize 的变动，新增下述代码（accumulate_grad_batches）
        # 计算需要的累积次数    目标总 Batch Size 为 32 原作者:(batchsize=8) * (devices=4)
        target_bs = 32
        current_bs = config['dataset_args']['train']['loader_args']['batch_size']
        acc_steps = max(1, target_bs // current_bs)

        if litmodule.global_rank == 0:
            print(f"Training with Batch Size: {current_bs}, Accumulate Grad Batches: {acc_steps}")
            print(f"Effective Batch Size: {current_bs * acc_steps}")
        # '''

        trainer = BaseTrainer(
            accelerator='gpu',
            strategy=st,
            enable_progress_bar=False,
            # max_epochs=config['exp_args']['epoch'],   # 原作者 真正意义上总共的轮数 配置文件中的epochs
            max_epochs=remaining_epochs,    # 【20251224 修改】使用剩余轮数
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
            num_nodes=num_nodes,
            check_val_every_n_epoch=check_val_every_n_epoch,
            devices=devices,
            accumulate_grad_batches=acc_steps  # <--- 仅需在这里新增这一行
        )
        trainer.fit(model=litmodule)
    else:
        st = 'ddp'
        litmodule.SetStrategy(st)
        trainer = BaseTrainer(
            accelerator='gpu',
            strategy=st,
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=False
        )
        trainer.validate(model=litmodule)
        if litmodule.global_rank == 0:
            print (litmodule.val_results)


class MyProgressCallback(pl.callbacks.Callback):
    def __init__(self, tqdm_total, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_total = tqdm_total

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            count = int(np.ceil(len(pl_module.train_dataloader_obj) / self.tqdm_total))
            # print(f"初始化进度条: {count} 步")
            self.myprogress = Tools.MyTqdm(range(count))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if pl_module.global_rank == 0: next(self.myprogress)

    def on_validation_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            tmp = len(pl_module.val_dataloader_obj)
            count = int(np.ceil(tmp / self.tqdm_total))
            self.myprogress = Tools.MyTqdm(range(count))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_rank == 0: next(self.myprogress)

# from pytorch_lightning.strategies import SingleDeviceStrategy
class BaseTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 通过降级PyTorch Lightning Lignting 2.x -> 1.9x 解决版本为问题
        if isinstance(self.strategy, pl.strategies.dp.DataParallelStrategy):
            tqdm_total = 1
        elif isinstance(self.strategy, pl.strategies.ddp.DDPStrategy):
            tqdm_total = self.num_nodes * self.num_devices
        else:
            raise NotImplementedError
        self.callbacks.append(MyProgressCallback(tqdm_total=tqdm_total))

        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.callbacks.append(lr_monitor)

# 2. Trainer 初始化时动态加载 (BaseTrainer.py)
# main.py 初始化 MM (即 SelfSupervisedLitModule)，进而调用父类 BaseLitModule 的 __init__。
class BaseLitModule(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.val_results = None
        self.save_hyperparameters(copy.deepcopy(config))
        # BaseLitModule.__init__ 调用 self.PrepareDataset()
        self.train_data, self.val_data = self.PrepareDataset()

        # 【20251219 新增】 初始化 epoch 偏移量，默认为 0：以适配我修改的模型加载逻辑
        self.epoch_offset = 0
    
    # PrepareDataset 方法内部使用了 Tools.rgetattr 来根据字符串名称获取类对象。
    def PrepareDataset(self):
        config = self.config
        # ① 获取类名字符串，即 'SelfSupervisedDataset'
        # ② Tools.rgetattr(Dataset, ...) 去 BiFusev2.Dataset 模块下找这个类
        train_datafunc = Tools.rgetattr(Dataset, config['dataset_args']['train']['dataset_type'])
        # ③ 实例化该类（这里就是调用 SelfSupervisedDataset(...) 的地方）
        train_data = train_datafunc(**config['dataset_args']['train'])
        
        val_datafunc = Tools.rgetattr(Dataset, config['dataset_args']['val']['dataset_type'])
        val_data = val_datafunc(**config['dataset_args']['val'])

        return train_data, val_data
    
    def SetStrategy(self, s):
        self.strategy = s
    
    # 【20251224 修改】 修改 TensorBoard 验证集曲线 (Eval)
    def WriteValResults(self, results):
        # 20251224 新增】 计算显示用的 epoch
        display_epoch = self.current_epoch + self.epoch_offset

        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            writer = self.logger.experiment
            #【20251224 修改】 加上偏移量，让 TensorBoard 的横坐标接续之前的训练
            for key, val in results.items():
                # writer.add_scalar('Eval/%s'%(key), val, self.current_epoch)   # 原作者
                # 【20251224 修改】 使用 display_epoch 替代 self.current_epoch
                writer.add_scalar('Eval/%s'%(key), val, display_epoch)

        elif isinstance(self.logger, pl_loggers.WandbLogger):
            # tmp = {'Eval/epoch': self.current_epoch}  # 原作者
            #【20251224 修改】使用 display_epoch 替代 self.current_epoch
            tmp = {'Eval/epoch': self.current_epoch + self.epoch_offset} 
            for key, val in results.items(): tmp['Eval/%s'%(key)] = val
            self.logger.experiment.log(tmp)
        else:
            raise ValueError('logger type weird')
    
    # 4.最终的数据加载 (DataLoader)
    # 实例化 Dataset 后，PyTorch Lightning 会调用 train_dataloader() 方法，该方法会调用 Dataset 的 CreateLoader 接口生成最终的 DataLoader。
    def train_dataloader(self):
        self.train_dataloader_obj = self.train_data.CreateLoader()
        return self.train_dataloader_obj
    
    def val_dataloader(self):
        self.val_dataloader_obj = self.val_data.CreateLoader()
        return self.val_dataloader_obj
    
    def configure_optimizers(self):
        optimizer_args = self.config['fitting_args']['optimizer_args']
        scheduler_args = self.config['fitting_args']['scheduler_args']
        optimizer_func = getattr(torch.optim, optimizer_args['type'])
        optimizer = optimizer_func(self.model.parameters(), **optimizer_args['args'])

        if scheduler_args is not None:
            scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_args['type'])
            scheduler = scheduler_func(optimizer, **scheduler_args['args'])
            print (optimizer, scheduler)
            return [optimizer], [scheduler]
        else: 
            print (optimizer)
            return [optimizer]
    '''
    # 原作者的模型加载逻辑    
    def on_train_epoch_start(self):
        if self.global_rank == 0: 
            print ('Epoch %d/%d'%(self.current_epoch, self.config['exp_args']['epoch']-1))
    '''
    # 【20251219 修改】 我理解的模型加载逻辑: 修改 终端打印的 Epoch
    def on_train_epoch_start(self):
        if self.global_rank == 0: 
            # 【20251224 新增】 计算当前显示的 epoch
            # PyTorch Lightning 内部的 self.current_epoch 每次运行都从 0 开始
            # 我们加上偏移量，使其显示为 "X+1" (因为 start_epoch 已经是 X+1 了)
            display_epoch = self.current_epoch + self.epoch_offset
            
            # 【20251225 新增】 硬停车逻辑 (Guardrail)
            # 如果显示的 Epoch 已经达到或超过配置的总 Epoch (例如 150 >= 150)
            # 说明已经跑完了规定的轮数，应该立即停止，防止出现 150/149
            if display_epoch >= self.config['exp_args']['epoch']:
                print(f"[Info] Reached target epoch {display_epoch}. Stopping training now.")
                self.trainer.should_stop = True
                return
            
            # 打印格式：Epoch 当前 / 总目标-1
            # 注意：config['exp_args']['epoch'] 是总轮数 (例如 200)
            print ('Epoch %d/%d'%(display_epoch, self.config['exp_args']['epoch']-1))
            
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            if self.val_results is not None:
                print (self.val_results)
                self.WriteValResults(self.val_results)
                # 【20251224 修改】 计算真实的 epoch 编号用于保存
                # self.model.Save(self.current_epoch, accuracy=-self.val_results['rmse'], replace=True)
                save_epoch = self.current_epoch + self.epoch_offset
                self.model.Save(save_epoch, accuracy=-self.val_results['rmse'], replace=True)
            else:
                print ('No val results!')
            self.val_results = None
    
    def after_validation_step(self, outputs):
        ## Need to call inside validation_step
        outputs = self.all_gather(outputs)
        if self.global_rank == 0:
            idx, pred, gt_depth = outputs
            idx = idx.flatten(0, 1).cpu().numpy()
            pred = pred.flatten(0, 1).cpu().numpy()
            gt_depth = gt_depth.flatten(0, 1).cpu().numpy()
            for i in range (pred.shape[0]):
                if self.val_idx_record[idx[i]] == 0:
                    acc = self.metrics_meters.update(pred[i], gt_depth[i])
                    self.val_idx_record[idx[i]] = 1

    def on_validation_epoch_start(self):
        if self.global_rank == 0:
            self.metrics_meters = Metric.MovingAverageEstimator(**self.config['metric_args'])
            self.val_idx_record = np.zeros(len(self.val_data), int)

    def validation_epoch_end(self, val_outs):
        if self.global_rank == 0:
            self.val_results = self.metrics_meters()

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Must implement
        """
    
    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Must implement
        """

    @abc.abstractmethod
    def write_logger(self, *args, **kwargs):
        pass
