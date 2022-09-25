import abc
import multiprocessing as mp
import os
import time
import typing
from dataclasses import dataclass
from queue import Empty, Full

import numpy as np
import torch

from .base_service import BaseService


@dataclass
class UpscalerQueueEntry:
    img:typing.Union[np.ndarray, torch.Tensor] = None
    step:int = 0

class BaseUpscalerService(BaseService):
    on_queue = None

    def __init__(self) -> None:
        super().__init__()

    def push_frame(self, img, step, timeout=10):
        self.job_queue.put(UpscalerQueueEntry(
            img=img, step=step
        ), timeout=timeout)
    
    #@abs.abstractmethod
    def proc_init(self):
        pass

    #@abs.abstractmethod
    def proc_job_recieved(self, job):
        img = job.img
        step = job.step
        img_up = self.upscale(img)
        entry = UpscalerQueueEntry(
            img=img_up, step=step
        )
        return entry

    #@abs.abstractmethod
    def proc_cleanup(self):
        pass
    
    #@abc.abstractmethod
    def upscale(self, img):
        pass
