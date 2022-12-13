import os, io
import sys
import threading
from typing import Dict, Tuple
import PIL
import torch.multiprocessing as mp
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
import flask as fl

from upscale.server.cache import DiskImageCache, MemoryImageCache
logging.basicConfig()

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
            logger.info('Upscaler started')
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

image_cache = MemoryImageCache()
# image_cache = DiskImageCache()

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

def upscaler_queue_handler():
    pipeline = get_pipeline()
    while True:
        entry = pipeline.result_queue.get() #type: UpscalerQueueEntry
        if entry.step in upscaler_queue_semas:
            sema = upscaler_queue_semas[entry.step] #type: threading.Semaphore
            if sema is not None:
                entry.frames = entry.frames.to('cpu', non_blocking=True)
                upscaler_queue_entries[entry.step] = entry
                sema.release()
        else:
            raise Exception('should not happen')

upscaler_queue_semas = {}
upscaler_queue_entries = {}
upscaler_queue_thread = None
if upscaler_queue_thread == None:
    upscaler_queue_thread = threading.Thread(target=upscaler_queue_handler, daemon=True)
    upscaler_queue_thread.start()

@blueprint.route('/image', methods=['POST'])
def upscale_image():
    global count, hitcount
    
    pre_scale = 1.0
    post_scale = 0.66
    
    count += 1
    profiler = Profiler()
    profiler.start('endpoint')
    
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
    # logger.debug(img.shape)
    
    profiler.start('endpoint.pipeline')
    upscaler = get_pipeline()
    profiler.end('endpoint.pipeline')
    
    profiler.start('endpoint.proc')
    if img.shape[0] * img.shape[1] > 1024*1024:
        pre_scale = 0.8
        post_scale = 0.85
    if img.shape[0] * img.shape[1] < 64*32:
        post_scale = 1.0
    if img.shape[0] * img.shape[1] > 4096*2048:
        logger.error(f"img is too big! {img.shape} > (4096x2048)")
        return fl.jsonify({
            'result':'err', 
            'err': f"img is too big! {img.shape} > (4096x2048)",
            'profiler': profiler.data
        })
    
    if pre_scale < 1.0:
        img = cv2.resize(img, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_AREA)
    
    my_sema = threading.Semaphore(0)
    upscaler_queue_semas[my_id] = my_sema
    
    try:
        upscaler.push_job(UpscalerQueueEntry(
            frames=torch.tensor(img, dtype=torch.uint8, device=upscaler.device).unsqueeze(0).clone(),
            audio_segment=None,
            step=my_id,
            elapsed=0,
            last_modified=time.time(),
            profiler=profiler,
        ), timeout=10)
    except (queue.Full, TimeoutError):
        logger.error('worker is busy? push timeout')
        return fl.jsonify({
            'result':'err', 
            'err': f'worker is busy',
            'profiler': profiler.data
        })
    
    try:
        my_sema.acquire(timeout=10)
    except TimeoutError:
        upscaler_queue_semas[my_id] = None
        upscaler_queue_entries[my_id] = None
        
        logger.error('worker is busy? wait timeout')
        return fl.jsonify({
            'result':'err', 
            'err': f'worker is busy',
            'profiler': profiler.data
        })
    upscaler_queue_semas[my_id] = None
    
    entry = upscaler_queue_entries[my_id] #type: UpscalerQueueEntry
    upscaler_queue_entries[my_id] = None
    
    assert entry.step == my_id
    profiler = entry.profiler
    profiler.end('endpoint.proc')
    
    profiler.start('endpoint.write')
    if(entry.frames.device != 'cpu'):
        entry.frames = entry.frames.cpu()
    frame = entry.frames.squeeze(0).numpy()
    if post_scale < 1.0:
        frame = cv2.resize(frame, None, fx=post_scale, fy=post_scale, interpolation=cv2.INTER_AREA)
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

    profiler.end('endpoint')
    return fl.jsonify({
        'result':'ok', 
        'url':img_path,
        'profiler': entry.profiler.data
    })

app = fl.Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=False, port=8088, use_reloader=False, threaded=True, host='0.0.0.0')
