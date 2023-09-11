import gc
import os, io
import sys
import threading
from typing import Dict, Tuple
import PIL
import torch.multiprocessing as mp
import logging
import torch, time, cv2
import numpy as np
from ...upscale.base_service import ProcessDeadException
from ...upscale.fsrcnn_upscaler import FsrcnnUpscalerService, UpscalerQueueEntry
from ...util.profiler import Profiler
import queue
import hashlib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import GifImagePlugin
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_ALWAYS
from PIL import Image
import flask as fl

from .stateful_cache import DiskImageCache, MemoryImageCache
logging.basicConfig()

logger = logging.getLogger("ImagePipeline")
logger.setLevel(logging.DEBUG)

blueprint = fl.Blueprint('upscale', __name__, url_prefix='/upscale')

upscaler_lock = mp.RLock()
upscaler: FsrcnnUpscalerService = None
upscaler_event = mp.Event()

def get_bytes_hash(buffer):
    return hashlib.sha1(buffer).hexdigest()

def pipeline_onqueue(entry:UpscalerQueueEntry):
    global upscaler
    upscaler.result_queue.put(UpscalerQueueEntry(
        frames=entry.frames.cpu().clone(),
        audio_segment=None,
        step=entry.step,
        elapsed=entry.elapsed,
        last_modified=entry.last_modified,
        profiler=entry.profiler
    ))

def get_pipeline():
    global upscaler
    start_pipeline()
    return upscaler

def start_pipeline():
    global upscaler
    with upscaler_lock:
        if upscaler is None:
            upscaler = FsrcnnUpscalerService(
                lr_level=3, device=0, denoising=False, denoise_rate=0.2, on_queue=pipeline_onqueue,
                upscaler_model='realesrgan', batch_size=1, jit_mode=False, lr_hr_resize=False
            )
            upscaler.exit_on_error = True
            upscaler.start()
            logger.info('Upscaler started')

def restart_pipeline():
    global upscaler
    with upscaler_lock:
        if upscaler is not None:
            assert not upscaler.proc.is_alive()
            upscaler = None
            
        start_pipeline()

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

def write_file(filename, buffer, profiler: Profiler):
    profiler.start('write_file.cache')
    image_cache.write_file(filename, buffer)
    profiler.end('write_file.cache')
    
    return f'/upscale/file/{filename}'

count = 0
hitcount = 0

def upscaler_queue_handler():
    global upscaler_queue_lock, upscaler_queue_semas, upscaler_queue_entries
    pipeline = get_pipeline()
    while True:
        entry = pipeline.result_queue.get() #type: UpscalerQueueEntry
        sema = None
        with upscaler_queue_lock:
            if entry.step in upscaler_queue_semas:
                sema = upscaler_queue_semas[entry.step] #type: threading.Semaphore
                if sema is not None:
                    entry.frames = entry.frames.to('cpu', non_blocking=True)
                    # gc.collect(generation=0)
                    # torch.cuda.empty_cache()
                    upscaler_queue_entries[entry.step] = entry
        if sema is not None:
            sema.release()
        else:
            raise Exception('should not happen')

upscaler_queue_lock = threading.RLock()
upscaler_queue_semas = {}
upscaler_queue_entries = {}
upscaler_queue_thread = None
if upscaler_queue_thread == None:
    upscaler_queue_thread = threading.Thread(target=upscaler_queue_handler, daemon=True)
    upscaler_queue_thread.start()

USE_CACHE = False

@blueprint.route('/image', methods=['POST'])
def upscale_image():
    global upscaler_queue_lock, upscaler_queue_semas, upscaler_queue_entries
    global count, hitcount
    
    pre_scale = 1.0
    post_scale = 0.66
    
    count += 1
    profiler = Profiler()
    profiler.start('endpoint')
    
    return_type = 'url'
    params = fl.request.args.to_dict()
    if 'return_type' in params:
        if params['return_type'] in ['url', 'file']:
            return_type = params['return_type']
        else:
            return fl.jsonify({
                'result':'err',
                'err': f'unknown return type {params["return_type"]}'
            }), 500
    
    profiler.start('endpoint.io.read')
    buffer = fl.request.files['file'].read()
    profiler.end('endpoint.io.read')
    profiler.start('endpoint.io.hash')
    my_id = get_bytes_hash(buffer)
    profiler.end('endpoint.io.hash')
    filename = my_id + '.png'
    
    if USE_CACHE:
        cache_path = has_file(filename)
        if cache_path is not None:
            logger.debug(f"return cached {filename} in {cache_path}")
            if return_type == 'url':
                return fl.jsonify({
                    'result':'ok',
                    'cache': 'hit',
                    'url':cache_path,
                    'profiler': profiler.data
                }), 200
            elif return_type == 'file':
                buffer = read_file(filename)
                if buffer is not None:
                    return fl.send_file(buffer, download_name=filename)
                else:
                    return fl.jsonify({
                        'result':'err',
                        'cache': 'miss',
                        'err': 'cache was hit but cleared before read all bytes'
                    }), 500
            else:
                raise Exception()
    
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
        }), 500
    
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
        }), 500
    
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
        }), 500
    
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
        }), 500
    
    if pre_scale < 1.0:
        img = cv2.resize(img, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_AREA)
    
    my_sema = threading.Semaphore(0)
    with upscaler_queue_lock:
        upscaler_queue_semas[my_id] = my_sema
    
    try:
        upscaler.push_job(UpscalerQueueEntry(
            frames=torch.tensor(img, dtype=torch.uint8, device=upscaler.device).unsqueeze(0),
            audio_segment=None,
            step=my_id,
            elapsed=0,
            last_modified=time.time(),
            profiler=profiler,
        ), timeout=20)
    except (queue.Full, TimeoutError):
        logger.error('worker is busy? push timeout')
        return fl.jsonify({
            'result':'err', 
            'err': f'worker is busy',
            'profiler': profiler.data
        }), 500
    except ProcessDeadException as ex:
        restart_pipeline()
        return fl.jsonify({
            'result':'err', 
            'err': f'worker is dead',
            'profiler': profiler.data
        }), 500
    
    try:
        my_sema.acquire(timeout=20)
    except TimeoutError:
        with upscaler_queue_lock:
            upscaler_queue_semas[my_id] = None
            upscaler_queue_entries[my_id] = None
        
        logger.error('worker is busy? wait timeout')
        return fl.jsonify({
            'result':'err', 
            'err': f'worker is busy',
            'profiler': profiler.data
        }), 500
        
    with upscaler_queue_lock:
        upscaler_queue_semas[my_id] = None
        
        if my_id in upscaler_queue_entries:
            entry = upscaler_queue_entries[my_id] #type: UpscalerQueueEntry
        else:
            logger.error('queue may interrupted. entry')
            return fl.jsonify({
                'result':'err', 
                'err': f'queue may interrupted. entry',
                'profiler': profiler.data
            }), 500
        upscaler_queue_entries[my_id] = None
    
    if entry is None:
        logger.error('queue may interrupted')
        return fl.jsonify({
            'result':'err', 
            'err': f'queue may interrupted',
            'profiler': profiler.data
        }), 500
    
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
    
    logger.debug(f'write_file: {filename}, img:{frame.shape}, {frame.dtype}')
    profiler.start('write_file.imsave')
    out_buffer = io.BytesIO()
    if frame.shape[-1] == 4:
        Image.fromarray(frame).save(out_buffer, format='PNG', optimize=False)
    else:
        Image.fromarray(frame).save(out_buffer, format='JPEG', progressive=True, quality=85, optimize=True)
    out_buffer.seek(0)
    profiler.end('write_file.imsave')
    
    if USE_CACHE:
        img_path = write_file(my_id+'.png', out_buffer, profiler)
        logger.debug(f'{frame.shape} {img_path}')
    profiler.end('endpoint.write.file')
    profiler.end('endpoint.write')

    profiler.end('endpoint')
    if return_type == 'url':
        assert USE_CACHE
        
        return fl.jsonify({
            'result':'ok', 
            'url':img_path,
            'profiler': entry.profiler.data
        }), 200
    elif return_type == 'file':
        out_buffer.seek(0)
        return fl.send_file(out_buffer, download_name=filename)
    else:
        raise Exception()

app = fl.Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=False, port=8088, use_reloader=False, threaded=True, host='0.0.0.0')
