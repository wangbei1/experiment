"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""
from datetime import datetime  # 正确的导入方式

from abc import ABC
from typing import Optional, Union, Dict
import pathlib
from torchinfo import summary

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from utils.lightning_utils import EMA
from .data_modules import BaseDataModule

torch.set_float32_matmul_precision("high")

from lightning.pytorch.callbacks import TQDMProgressBar

class LRTQDMProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # 调用父类获取已有指标
        items = super().get_metrics(trainer, model)
        # 获取当前学习率
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            items['lr'] = lr
        return items


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger if logger else False
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )

class ProgressBarMetricsLogger(pl.Callback):
    """
    在每次验证结束后，抓一份'进度条里正在显示的指标'，
    追加写到 loss_log_path 里（一行）。
    这依赖于你定义的 LRTQDMProgressBar.get_metrics().
    """

    def __init__(self, loss_log_path: pathlib.Path, progress_bar: LRTQDMProgressBar):
        super().__init__()
        self.loss_log_path = loss_log_path
        self.progress_bar = progress_bar  # 我们需要用它的get_metrics逻辑保证一致

    def on_validation_end(self, trainer, pl_module):
        """
        Lightning会在每轮验证loop结束后调用这个函数。
        这时 trainer 里已经有最新的 prediction/fvd、psnr 等指标，
        以及我们在 LRTQDMProgressBar 里的 lr 注入逻辑。
        """

        # 让我们复用你自己定义的进度条的 metrics 计算逻辑
        try:
            bar_metrics = self.progress_bar.get_metrics(trainer, pl_module)
        except Exception as e:
            print(f"⚠️ 进度条metrics收集失败: {e}")
            bar_metrics = {}

        # bar_metrics 是个 dict，比如:
        # {
        #   "prediction/fvd": tensor(1499.0),
        #   "prediction/is": 4.150,
        #   ...,
        #   "lr": 2.2e-6,
        #   "v_num": "kitm",
        #   "epoch": 0,
        #   "step": 44,
        #   ...
        # }

        # 把 tensor 转成 python float / str
        serializable_items = {}
        for k, v in bar_metrics.items():
            if torch.is_tensor(v):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            serializable_items[k] = v

        # 我们只想把有用的 metrics 打印进去，通常可以过滤一下Lightning自带的一些字段
        # 比如 'v_num', 'epoch', 'step' 也许你想保留，也可以保留全部，这里我们保留全部但排序一下
        ordered_keys = sorted(serializable_items.keys())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line_parts = [f"{timestamp}"]
        for k in ordered_keys:
            line_parts.append(f"{k}={serializable_items[k]}")
        line = " | ".join(line_parts) + "\n"

        # 追加写入日志文件
        try:
            with open(self.loss_log_path, "a") as f:
                f.write(line)
            print(f"✅ 进度条指标已写入 {self.loss_log_path}")
        except Exception as e:
            print(f"❌ 写入 {self.loss_log_path} 失败: {e}")

class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = self.data_module_cls(root_cfg, self.compatible_datasets)
        # 创建 loss 日志文件
        self.loss_log_path = pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "loss.txt"
        self._init_loss_log_file()
    
    def _build_common_callbacks(self):
        return [EMA(**self.cfg.ema)]
    
    def _init_loss_log_file(self):
        """初始化 loss 日志文件"""
        # 创建文件并写入表头

        print(f"Loss日志文件创建在: {self.loss_log_path}")


    def _write_model_summary_to_file(self, model_summary_str):
        """将模型摘要写入 loss.txt 文件"""
        try:
            # 读取现有内容
            if self.loss_log_path.exists():
                with open(self.loss_log_path, 'r') as f:
                    existing_content = f.read()
            else:
                existing_content = ""
            
            # 创建新的内容：模型摘要 + 现有内容
            new_content = self._format_model_summary(model_summary_str) + "\n\n" + existing_content
            
            # 写回文件
            with open(self.loss_log_path, 'w') as f:
                f.write(new_content)
                
        except Exception as e:
            print(f"❌ 写入模型摘要失败: {e}")

    def _format_model_summary(self, model_summary_str):
        """格式化模型摘要，使其更美观"""
        formatted_summary = (
            "=" * 80 + "\n"
            "🧠 MODEL ARCHITECTURE SUMMARY\n"
            "=" * 80 + "\n\n"
            f"Model Class: {type(self.algo).__name__}\n"
            f"Input Size: {getattr(self.algo, 'x_shape', 'N/A')}\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "─" * 80 + "\n\n"
            f"{model_summary_str}\n"
            "─" * 80 + "\n"
            "📊 LOSS RECORDS (below)\n"
            "=" * 80 + "\n"
        )
        return formatted_summary
    



    def training(self) -> None:
        """
        All training happens here
        """

        # ----------------------------
        # 常量：全都写死在这里，想调直接改数字
        # ----------------------------
        CHECKPOINT_EVERY_N_STEPS = 5000      # 每多少个训练 step 存一次 ckpt
        VAL_EVERY_N_STEPS = 5000            # 每多少个训练 step 跑一次验证
        TRAIN_MAX_STEPS = 200000            # 训练最多跑多少个优化 step
        DATALOADER_RELOAD_EVERY_N_EPOCHS = 0  # 按 step 训练就别每个 epoch 重建 dataloader

        if not self.algo:
            self.algo = self._build_algo()
        self.algo.loss_log_path = self.loss_log_path

        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        # 如果你想导出模型结构摘要，可以把下面这段解开
        # x = torch.rand(1, 8, 3, 256, 256, dtype=torch.float32)
        # external_cond = torch.rand(1, 8, 180, 256, 256, dtype=torch.float32)
        # input_data = [x, external_cond, torch.full((1, 8), 0.5, dtype=torch.float32)]
        # model_summary_str = summary(
        #     self.algo.diffusion_model,
        #     input_data=input_data,
        #     depth=7,
        #     col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"),
        #     verbose=0,
        # )
        # self._write_model_summary_to_file(model_summary_str)

        # ----------------------------
        # callbacks
        # ----------------------------
        callbacks = [LearningRateMonitor(logging_interval='step')]

        if self.logger:
            progress_bar_cb = LRTQDMProgressBar(refresh_rate=1)
            callbacks.append(progress_bar_cb)
        else:
            progress_bar_cb = LRTQDMProgressBar(refresh_rate=1)
            # 即使没logger我还是构造一个，后面要用get_metrics
            # 也可以按你喜好写成两支分支

        checkpoint_callback = ModelCheckpoint(
            dirpath=(
                pathlib.Path(
                    hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
                )
                / "checkpoints"
            ),
            filename="dfot-{step}",
            every_n_train_steps=CHECKPOINT_EVERY_N_STEPS,
            every_n_epochs=None,
            save_top_k=-1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # EMA等
        callbacks += self._build_common_callbacks()

        # ✅ 新增: 把进度条上的信息写到 loss_log_path 的记录器
        callbacks.append(ProgressBarMetricsLogger(self.loss_log_path, progress_bar_cb))

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            precision=self.cfg.training.precision,
            detect_anomaly=False,

            val_check_interval=VAL_EVERY_N_STEPS,
            check_val_every_n_epoch=None,
            limit_val_batches=self.cfg.validation.limit_batch,
            num_sanity_val_steps=0,

            max_steps=TRAIN_MAX_STEPS,
            max_epochs=None,
            max_time=None,

            reload_dataloaders_every_n_epochs=DATALOADER_RELOAD_EVERY_N_EPOCHS,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
        )

        trainer.fit(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.test.inference_mode,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )
