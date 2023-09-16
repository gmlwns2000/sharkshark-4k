import abc
import torch.multiprocessing as mp
import os
import time
import typing
from dataclasses import dataclass
from queue import Empty, Full

import numpy as np
import torch

from ..util.profiler import Profiler

from .base_service import BaseService


@dataclass
class UpscalerQueueEntry:
    frames:torch.Tensor = None
    audio_segment:torch.Tensor = None
    step:int = 0
    elapsed:float = 0
    last_modified:float = 0
    profiler: Profiler = None

class BaseUpscalerService(BaseService):
    profiler: Profiler
    lr_shape = (720, 1280)
    output_shape = (1440, 2560)
    on_queue = None

    def __init__(self) -> None:
        super().__init__()
    
    #@abs.abstractmethod
    def proc_init(self):
        pass

    #@abs.abstractmethod
    def proc_job_recieved(self, job: UpscalerQueueEntry):
        self.profiler = job.profiler
        
        t = time.time()
        job.profiler.end('recoder.output')
        frames = job.frames
        job.profiler.start('upscaler.upscale')
        if frames is not None:
            frames_up = self.upscale(frames)
        else:
            frames_up = None
        job.profiler.end('upscaler.upscale')
        elapsed = time.time() - t
        job.profiler.start('upscaler.output')
        entry = UpscalerQueueEntry(
            frames=frames_up, step=job.step, audio_segment=job.audio_segment, 
            elapsed=elapsed, last_modified=time.time(), profiler=job.profiler
        )
        return entry

    #@abs.abstractmethod
    def proc_cleanup(self):
        pass
    
    #@abc.abstractmethod
    def upscale(self, frames):
        pass
