import os, io
import sys
import threading
from typing import Dict, Tuple
import PIL
import flask as fl
import multiprocessing as mp
import logging
import torch, time, cv2
import numpy as np
from upscale.fsrcnn_upscaler import FsrcnnUpscalerService, UpscalerQueueEntry
from util.profiler import Profiler
import queue
import hashlib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import GifImagePlugin
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_ALWAYS
from PIL import Image

logger = logging.getLogger("ImagePipeline")
logger.setLevel(logging.DEBUG)
blueprint = fl.Blueprint('upscale', __name__, url_prefix='/upscale')

upscaler_lock = mp.Lock()
upscaler = None
upscaler_event = mp.Event()

def get_bytes_hash(buffer):
    return hashlib.sha1(buffer).hexdigest()

def get_pipeline():
    global upscaler
    with upscaler_lock:
        if upscaler is None:
            upscaler = FsrcnnUpscalerService(
                lr_level=3, device=0, denoising=False, denoise_rate=0.2, 
                upscaler_model='realesrgan', batch_size=1, jit_mode=False, lr_hr_resize=False
            )
            upscaler.exit_on_error = True
            upscaler.start()
            logging.info('Upscaler started')
    return upscaler

@blueprint.route('/ping')
def ping():
    return 'pong'

@blueprint.route('/file/<filename>')
def upload_file(filename:str):
    if filename.find('..') >= 0 or filename.find('/') >= 0 or filename.find('~') >= 0 or filename.find('$') >= 0 or filename.find('%') >= 0:
        return {
            'status': 'err',
            'err': f'forbidden path {filename}'
        }, 500
    
    buffer = read_file(filename)
    if buffer is None:
        return {
            'status': 'err',
            'err': 'file not found'
        }, 404
    
    return fl.send_file(buffer, download_name=filename)

class ImageCache:
    def has_file(self, filename:str):
        pass
    
    def read_file(self, filename:str):
        pass
    
    def write_file(self, filename:str, buffer:io.BytesIO):
        pass

class DiskImageCache(ImageCache):
    def has_file(self, filename):
        if os.path.exists(f'./cache/{filename}'):
            return f'/upscale/file/{filename}'
        else:
            return None
    
    def read_file(self, filename):
        file_path = f'./cache/{filename}'
        if os.path.exists(file_path):
            with open(f'./cache/{filename}', 'rb') as f:
                return io.BytesIO(f.read())
        else:
            return None
    
    def write_file(self, filename, buffer):
        os.makedirs('./cache', exist_ok=True)
        file_path = f'./cache/{filename}'
        with open(file_path, 'wb') as f:
            f.write(buffer)

class WithWrapper:
    def __init__(self, enter, exit) -> None:
        self.enter = enter
        self.exit = exit
    
    def __enter__(self):
        self.enter()
 
    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

class ReaderWriterObject:
    def __init__(self, object) -> None:
        self.object = object
        self.mutex = threading.RLock()
        self.wrt = threading.RLock()
        self.nreader = 0
    
    def get(self): return self.object
    def set(self, v): self.object = v
    
    def read(self):
        return WithWrapper(self.start_read, self.end_read)
    
    def start_read(self):
        self.mutex.acquire()
        self.nreader += 1
        if (self.nreader == 1):
            self.wrt.acquire()
        self.mutex.release()
        
    def end_read(self):
        self.mutex.acquire()
        self.nreader -= 1
        assert self.nreader >= 0
        if (self.nreader == 0):
            self.wrt.release()
        self.mutex.release()
    
    def write(self):
        return WithWrapper(self.start_write, self.end_write)
    
    def start_write(self):
        self.wrt.acquire()
    
    def end_write(self):
        self.wrt.release()

class MemoryImageCache:
    def __init__(self, cache_size_bytes=1024*1024*1024*4) -> None:
        self.max_size = cache_size_bytes #1020*1024*8
        self.size = 0
        self._bank: Dict[str, Tuple[float, ReaderWriterObject]] = {} #filename -> (last_tick, mutex, wrt, num_readers, io.BytesIO)
        self.bank = ReaderWriterObject(self._bank)
    
    def contains(self, key):
        with self.bank.read():
            has_key = key in self.bank.get()
            # logger.debug(f'contains {key} {has_key}')
            return has_key
    
    def sizeof(self, key):
        with self.bank.read():
            if not self.contains(key): return 0
            item = self.bank.get()[key][-1] #type: ReaderWriterObject
            with item.read():
                buffer = item.get() #type: io.BytesIO
                return sys.getsizeof(buffer)
    
    def remove(self, key):
        if not self.contains(key): return
        with self.bank.write():
            size = self.sizeof(key)
            self.size -= size
            item = self.bank.get()[key]
            with item[-1].write():
                del self.bank.get()[key]
        
        logger.debug(f'removed {self.size / self.max_size * 100} %')
    
    def add(self, key, value: io.BytesIO):
        if self.contains(key): raise Exception('already exists')
        
        with self.bank.write():
            self.size += sys.getsizeof(value)
            if self.size > self.max_size:
                logger.debug(f'need to evict {self.size} {self.max_size}')
                self.evict(self.size - self.max_size)
            
            self.bank.get()[key] = (time.time(), ReaderWriterObject(value))
        
        logger.debug(f'added {self.size / self.max_size * 100} %')
    
    def cloneof(self, key):
        with self.bank.read():
            if not self.contains(key): raise Exception('not found')
            item = self.bank.get()[key]
            with item[-1].read():
                self.bank.get()[key] = (time.time(), item[-1])
                buffer = item[-1].get() #type: io.BytesIO
                return io.BytesIO(buffer.getvalue())
    
    def evict(self, to_bytes):
        #threadsafe here
        logger.debug(f'need to evict {to_bytes} bytes')
        while to_bytes > 0 and len(self.bank.get()) > 0:
            minval = 987654312340
            minidx = None
            for key, value in self.bank.get().items():
                if value[0] < minval:
                    minval = value[0]
                    minidx = key
            if minidx is not None:
                size = self.sizeof(minidx)
                to_bytes -= size
                logger.debug(f'evict {size} bytes')
                self.remove(minidx)
            else:
                return
    
    # interfaces
    def has_file(self, filename:str):
        if self.contains(filename):
            return f'/upscale/file/{filename}'
        else: return None
    
    def read_file(self, filename:str):
        if self.contains(filename):
            return self.cloneof(filename)
        else:
            return None
    
    def write_file(self, filename:str, buffer:io.BytesIO):
        self.add(filename, buffer)

image_cache = MemoryImageCache()

def has_file(filename):
    return image_cache.has_file(filename)

def read_file(filename):
    return image_cache.read_file(filename)

def write_file(filename, img, profiler: Profiler):
    logger.debug(f'write_file: {filename}, img:{img.shape}, {img.dtype}')
    profiler.start('write_file.imsave')
    buffer = io.BytesIO()
    if img.shape[-1] == 4:
        Image.fromarray(img).save(buffer, format='PNG', optimize=False)
    else:
        Image.fromarray(img).save(buffer, format='JPEG', progressive=True, quality=85, optimize=True)
    buffer.seek(0)
    profiler.end('write_file.imsave')
    
    profiler.start('write_file.cache')
    image_cache.write_file(filename, buffer)
    profiler.end('write_file.cache')
    
    return f'/upscale/file/{filename}'

count = 0
hitcount = 0

@blueprint.route('/image', methods=['POST'])
def upscale_image():
    global count, hitcount
    count += 1
    profiler = Profiler()
    
    profiler.start('endpoint.io.read')
    buffer = fl.request.files['file'].read()
    profiler.end('endpoint.io.read')
    profiler.start('endpoint.io.hash')
    my_id = get_bytes_hash(buffer)
    profiler.end('endpoint.io.hash')
    filename = my_id + '.png'
    
    cache_path = has_file(filename)
    if cache_path is not None:
        hitcount += 1
        logger.debug(f"return cached {filename} in {cache_path}, hitratio: {hitcount/count*100:.5f}%")
        return fl.jsonify({
            'result':'ok', 
            'cache': 'hit',
            'url':cache_path,
            'profiler': profiler.data
        })
    
    profiler.start('endpoint.io.imdecode')
    buffer = io.BytesIO(buffer)
    try:
        pil_img = Image.open(buffer)
        if not pil_img.mode in ['RGB', 'RGBA']:
            if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
                pil_img = pil_img.convert('RGBA')
            else:
                pil_img = pil_img.convert('RGB')
        img = np.asarray(pil_img)
        logger.debug(f'upscale origianl shape {img.shape}')
        # aa = Image.open(buffer)
        # aa.save('hello.jpg')
    except PIL.UnidentifiedImageError:
        img = None
    profiler.end('endpoint.io.imdecode')
    
    if img is None:
        logger.error(f"img is none. did you give correct image blob?")
        return fl.jsonify({
            'result':'err', 
            'err': 'img is none. did you give correct image blob?',
            'profiler': profiler.data
        })
    
    is_mono = False
    if len(img.shape) == 2:
        is_mono = True
        img = np.repeat(img.reshape((img.shape[0], img.shape[1], 1)), 3, axis=-1)
    
    if len(img.shape) != 3:
        logger.error(f"img must be 3D but got {img.shape}")
        return fl.jsonify({
            'result':'err', 
            'err': f'img must be 3D but got {img.shape}',
            'profiler': profiler.data
        })
    
    alpha_map = None
    if img.shape[-1] == 4:
        alpha_map = img[:,:,-1]
        img = img[:,:,:3]
        logger.debug(f'upscale alpha map {alpha_map.shape}')
    
    if img.shape[-1] != 3:
        logger.error(f"img must be RGB or RGBA but got {img.shape}")
        return fl.jsonify({
            'result':'err', 
            'err': f'img must be RGB or RGBA but got {img.shape}',
            'profiler': profiler.data
        })
    
    assert img.shape[-1] == 3
    logger.debug(img.shape)
    
    profiler.start('endpoint.pipeline')
    upscaler = get_pipeline()
    profiler.end('endpoint.pipeline')
    
    profiler.start('endpoint.proc')
    upscaler.push_job(UpscalerQueueEntry(
        frames=torch.tensor(img, dtype=torch.uint8, device=upscaler.device).unsqueeze(0),
        audio_segment=torch.empty((1,)),
        step=my_id,
        elapsed=0,
        last_modified=time.time(),
        profiler=profiler,
    ), timeout=3600)
    
    #wait for id is finished
    while True:
        try:
            entry = upscaler.get_result() #type: UpscalerQueueEntry
            logger.debug(f'processed id {entry.step}')
            if entry.step == my_id:
                profiler = entry.profiler
                break
            else:
                upscaler.result_queue.put(entry)
                time.sleep(0.01)
            #     upscaler_event.set()
            # upscaler_event.wait(timeout=0.5)
            # upscaler_event.clear()
        except TimeoutError:
            time.sleep(0.01)
            pass
    profiler.end('endpoint.proc')
    
    profiler.start('endpoint.write')
    if(entry.frames.device != 'cpu'):
        entry.frames = entry.frames.cpu()
    frame = entry.frames.squeeze(0).numpy()
    frame = cv2.resize(frame, None, fx=0.66, fy=0.66, interpolation=cv2.INTER_AREA)
    if alpha_map is not None:
        alpha_map = cv2.resize(alpha_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = np.concatenate([frame, alpha_map.reshape((frame.shape[0], frame.shape[1], 1))], axis=-1)
    if is_mono:
        #TODO: handle mono image correctly...
        frame = frame
    profiler.start('endpoint.write.file')
    img_path = write_file(my_id+'.png', frame, profiler)
    profiler.end('endpoint.write.file')
    logger.debug(f'{frame.shape} {img_path}')
    profiler.end('endpoint.write')
    
    return fl.jsonify({
        'result':'ok', 
        'url':img_path,
        'profiler': entry.profiler.data
    })

if __name__ == '__main__':
    app = fl.Flask(__name__)
    app.register_blueprint(blueprint)
    
    app.run(debug=False, port=8088, use_reloader=False, threaded=True, host='0.0.0.0')
