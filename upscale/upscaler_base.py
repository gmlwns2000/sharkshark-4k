import abc
import torch.multiprocessing as mp
import os
import time
import typing
from dataclasses import dataclass
from queue import Empty, Full

import numpy as np
import torch

from util.profiler import Profiler

from .base_service import BaseService


@dataclass
class UpscalerQueueEntry:
    frames:typing.Union[np.ndarray, torch.Tensor] = None
    audio_segment:np.ndarray = None
    step:int = 0
    elapsed:float = 0
    last_modified:float = 0
    profiler: Profiler = None

class BaseUpscalerService(BaseService):
    on_queue = None

    def __init__(self) -> None:
        super().__init__()
    
    #@abs.abstractmethod
    def proc_init(self):
        pass

    #@abs.abstractmethod
    def proc_job_recieved(self, job: UpscalerQueueEntry):
        t = time.time()
        job.profiler.end('recoder.output')
        frames = job.frames
        job.profiler.start('upscaler.upscale')
        frames_up = self.upscale(frames)
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
