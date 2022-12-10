#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This file contains the classes used to send videostreams to Twitch
"""
from __future__ import print_function, division
import numpy as np
import subprocess
import signal, io
import threading
import sys
import torch
try:
    import Queue as queue
except ImportError:
    import queue
import time
import os

import requests

AUDIORATE = 44100


class TwitchOutputStream(object):
    """
    Initialize a TwitchOutputStream object and starts the pipe.
    The stream is only started on the first frame.

    :param twitch_stream_key:
    :type twitch_stream_key:
    :param width: the width of the videostream (in pixels)
    :type width: int
    :param height: the height of the videostream (in pixels)
    :type height: int
    :param fps: the number of frames per second of the videostream
    :type fps: float
    :param enable_audio: whether there will be sound or not
    :type enable_audio: boolean
    :param ffmpeg_binary: the binary to use to create a videostream
        This is usually ffmpeg, but avconv on some (older) platforms
    :type ffmpeg_binary: String
    :param verbose: show ffmpeg output in stdout
    :type verbose: boolean
    """
    def __init__(self,
            twitch_stream_key,
            width=640,
            height=480,
            fps=30.,
            ffmpeg_binary="ffmpeg",
            enable_audio=False,
            verbose=False,
            output_file=None,
        ):
        self.twitch_stream_key = twitch_stream_key
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_process = None
        self.audio_pipe = None
        self.ffmpeg_binary = ffmpeg_binary
        self.verbose = verbose
        self.audio_enabled = enable_audio
        self.output_file = output_file
        try:
            self.reset()
        except OSError:
            print("There seems to be no %s available" % ffmpeg_binary)
            if ffmpeg_binary == "ffmpeg":
                print("ffmpeg can be installed using the following"
                      "commands")
                print("> sudo add-apt-repository "
                      "ppa:mc3man/trusty-media")
                print("> sudo apt-get update && "
                      "sudo apt-get install ffmpeg")
            sys.exit(1)
        self.lock = threading.Lock()

    def check_proc(self):
        with self.lock:
            if self.ffmpeg_process is None:
                return self.reset()
            ret_code = self.ffmpeg_process.poll()
            if ret_code != None:
                print("FFMPEG IS FAILED!!!")
                os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                exit(-1)
                if self.ffmpeg_process is not None:
                # Close the previous stream
                    try:
                        self.ffmpeg_process.send_signal(signal.SIGTERM)
                    except OSError:
                        pass
                self.ffmpeg_process = None
                print('TwitchOutputStream: FFMPEG DEAD!', ret_code)
                time.sleep(1)
                print('TwitchOutputStream: FFMPEG RETRY!', ret_code)
                return self.reset()
            
    
    def reset(self):
        """
        Reset the videostream by restarting ffmpeg
        """

        if self.ffmpeg_process is not None:
            # Close the previous stream
            try:
                self.ffmpeg_process.send_signal(signal.SIGINT)
            except OSError:
                pass

        command = []
        command.extend([
            self.ffmpeg_binary,
            '-loglevel', 'verbose',
            '-y',       # overwrite previous file/stream
            # '-re',    # native frame-rate
            '-analyzeduration', '1',
            '-f', 'rawvideo',
            '-r', '%d' % self.fps,  # set a fixed frame rate
            '-vcodec', 'rawvideo',
            # size of one frame
            '-s', '%dx%d' % (self.width, self.height),
            '-pix_fmt', 'rgb24',  # The input are raw bytes
            '-thread_queue_size', str(4096),
            '-i', '-',  # The input comes from a pipe

            # Twitch needs to receive sound in their streams!
            # '-an',            # Tells FFMPEG not to expect any audio
        ])
        if self.audio_enabled:
            command.extend([
                '-ar', '%d' % AUDIORATE,
                '-ac', '2',
                '-f', 's16le',
                '-thread_queue_size', str(4096),
                '-i', '/tmp/audiopipe'
            ])
        else:
            command.extend([
                '-f', 'lavfi',
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100'
            ])
        bitrate = '24000k'
        command.extend([
            # VIDEO CODEC PARAMETERS
            # -profile:v high444p
            *(f'-bufsize:v 100M -c:v h264_nvenc -cq 23 -preset p7 -spatial-aq 1 -temporal-aq 1 -b_ref_mode 2 -bf 4 -rc-lookahead 20'.split()),
            # *(f'-vcodec libx264 -b:v {bitrate} -minrate:v {bitrate} -maxrate:v {bitrate} -bufsize:v {bitrate} -preset medium -crf 16 -pix_fmt yuv420p'.split()),
            '-r', '%d' % self.fps,
            '-s', '%dx%d' % (self.width, self.height),
            # '-bufsize', bitrate,
            '-g', str(int(self.fps*2)),     # key frame distance

            # AUDIO CODEC PARAMETERS
            '-acodec', 'aac', '-strict', '-2', #'-ar', '44100', '-b:a', '320k',
            '-bufsize', '320k',
            #'-ac', '1',

            # MAP THE STREAMS
            # use only video from first input and only audio from second
            '-map', '0:v', '-map', '1:a',

            # NUMBER OF THREADS
            '-threads', '32',

            # STREAM TO TWITCH
            '-f', 'flv', '-flvflags', 'no_duration_filesize', 
            self.output_file if self.output_file is not None else self.get_closest_ingest(),
        ])

        devnullpipe = subprocess.DEVNULL
        if self.verbose:
            devnullpipe = None
        # devnullpipe = subprocess.DEVNULL
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = "1"
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=devnullpipe,
            stdout=devnullpipe,
            env=my_env,
            bufsize=1024*1024
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # sigint so avconv can clean up the stream nicely
        self.ffmpeg_process.send_signal(signal.SIGINT)
        # waiting doesn't work because of reasons I don't know
        # self.pipe.wait()

    def send_video_frame(self, frame):
        """Send frame of shape (height, width, 3)
        with values between 0 and 1.
        Raises an OSError when the stream is closed.

        :param frame: array containing the frame.
        :type frame: numpy array with shape (height, width, 3)
            containing values between 0.0 and 1.0
        """
        
        self.check_proc()

        assert frame.shape == (self.height, self.width, 3)

        if frame.dtype == np.uint8 or frame.dtype == torch.uint8:
            pass
        else:
            frame = np.clip(255*frame, 0, 255).astype('uint8')
        try:
            if isinstance(frame, np.ndarray):
                self.ffmpeg_process.stdin.write(frame.tobytes())
            elif isinstance(frame, torch.Tensor):
                # buff = io.BytesIO()
                # torch.save(frame, buff)
                # buff.seek(0)
                # self.ffmpeg_process.stdin.write(buff.read())
                self.ffmpeg_process.stdin.write(frame.numpy().tostring())
            else: raise Exception('unknown frame type')
        except OSError:
            # The pipe has been closed. Reraise and handle it further
            # downstream
            raise

    def send_audio(self, left_channel, right_channel):
        """Add the audio samples to the stream. The left and the right
        channel should have the same shape.
        Raises an OSError when the stream is closed.

        :param left_channel: array containing the audio signal.
        :type left_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        :param right_channel: array containing the audio signal.
        :type right_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        """
        
        self.check_proc()
        
        if self.audio_pipe is None:
            if not os.path.exists('/tmp/audiopipe'):
                os.mkfifo('/tmp/audiopipe')
            self.audio_pipe = os.open('/tmp/audiopipe', os.O_WRONLY)

        assert len(left_channel.shape) == 1
        assert left_channel.shape == right_channel.shape

        frame = np.column_stack((left_channel, right_channel)).flatten()

        frame = np.clip(32767*frame, -32767, 32767).astype('int16')
        try:
            os.write(self.audio_pipe, frame.tostring())
        except OSError:
            # The pipe has been closed. Reraise and handle it further
            # downstream
            raise

    def get_closest_ingest(self):
        closest_server = requests.get(url='https://ingest.twitch.tv/api/v2/ingests').json()['ingests'][0]
        url_template = closest_server['url_template']
        print("Streaming to closest server: %s at %s" % (closest_server['name'],
                                                         url_template.replace('/app/{stream_key}', '')))
        return url_template.format(
            stream_key=self.twitch_stream_key)


class TwitchOutputStreamRepeater(TwitchOutputStream):
    """
    This stream makes sure a steady framerate is kept by repeating the
    last frame when needed.

    Note: this will not generate a stable, stutter-less stream!
     It does not keep a buffer and you cannot synchronize using this
     stream. Use TwitchBufferedOutputStream for this.
    """
    def __init__(self, *args, **kwargs):
        super(TwitchOutputStreamRepeater, self).__init__(*args, **kwargs)

        self.lastframe = np.ones((self.height, self.width, 3))
        self._send_last_video_frame()   # Start sending the stream

        if self.audio_enabled:
            # some audible sine waves
            xl = np.linspace(0.0, 10*np.pi, int(AUDIORATE/self.fps) + 1)[:-1]
            xr = np.linspace(0.0, 100*np.pi, int(AUDIORATE/self.fps) + 1)[:-1]
            self.lastaudioframe_left = np.sin(xl)
            self.lastaudioframe_right = np.sin(xr)
            self._send_last_audio()   # Start sending the stream

    def _send_last_video_frame(self):
        try:
            super(TwitchOutputStreamRepeater,
                  self).send_video_frame(self.lastframe)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            pass
        else:
            # send the next frame at the appropriate time
            threading.Timer(1./self.fps,
                            self._send_last_video_frame).start()

    def _send_last_audio(self):
        try:
            super(TwitchOutputStreamRepeater,
                  self).send_audio(self.lastaudioframe_left,
                                   self.lastaudioframe_right)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            pass
        else:
            # send the next frame at the appropriate time
            threading.Timer(1./self.fps,
                            self._send_last_audio).start()

    def send_video_frame(self, frame):
        """Send frame of shape (height, width, 3)
        with values between 0 and 1.

        :param frame: array containing the frame.
        :type frame: numpy array with shape (height, width, 3)
            containing values between 0.0 and 1.0
        """
        self.lastframe = frame

    def send_audio(self, left_channel, right_channel):
        """Add the audio samples to the stream. The left and the right
        channel should have the same shape.

        :param left_channel: array containing the audio signal.
        :type left_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        :param right_channel: array containing the audio signal.
        :type right_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        """
        self.lastaudioframe_left = left_channel
        self.lastaudioframe_right = right_channel

BUFFER_QSIZE = 64

class TwitchBufferedOutputStream(TwitchOutputStream):
    """
    This stream makes sure a steady framerate is kept by buffering
    frames. Make sure not to have too many frames in buffer, since it
    will increase the memory load considerably!

    Adding frames is thread safe.
    """
    def __init__(self, *args, **kwargs):
        super(TwitchBufferedOutputStream, self).__init__(*args, **kwargs)
        self.last_frame = np.ones((self.height, self.width, 3))
        self.last_frame_time = None
        self.next_video_send_time = None
        self.frame_counter = 0
        self.q_video = queue.PriorityQueue(maxsize=BUFFER_QSIZE)

        # don't call the functions directly, as they block on the first
        # call
        self.t = threading.Timer(0.0, self._send_video_frame)
        self.t.daemon = True
        self.t.start()

        if self.audio_enabled:
            # send audio at about the same rate as video
            # this can be changed
            self.last_audio = (np.zeros((int(AUDIORATE/self.fps), )),
                               np.zeros((int(AUDIORATE/self.fps), )))
            self.last_audio_time = None
            self.next_audio_send_time = None
            self.audio_frame_counter = 0
            self.q_audio = queue.PriorityQueue(maxsize=BUFFER_QSIZE)
            self.t = threading.Timer(0.0, self._send_audio)
            self.t.daemon = True
            self.t.start()

    def _send_video_frame(self):
        start_time = time.time()
        try:
            frame = self.q_video.get_nowait()
            # frame[0] is frame count of the frame
            # frame[1] is the frame
            frame = frame[1]
        except IndexError:
            frame = self.last_frame
            frame = None
        except queue.Empty:
            frame = self.last_frame
            frame = None
        else:
            self.last_frame = frame

        try:
            if frame is not None:
                super(TwitchBufferedOutputStream, self
                    ).send_video_frame(frame)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            # Don't call this function again and everything should be
            # cleaned up just fine.
            return

        # send the next frame at the appropriate time
        FASTER_SLEEP = 1.0
        if self.next_video_send_time is None:
            self.t = threading.Timer(1./self.fps * FASTER_SLEEP, self._send_video_frame)
            self.next_video_send_time = start_time + 1./self.fps
        else:
            self.next_video_send_time += 1./self.fps
            next_event_time = (self.next_video_send_time - start_time) * FASTER_SLEEP
            if next_event_time > 0:
                self.t = threading.Timer(next_event_time,
                                         self._send_video_frame)
            else:
                # we should already have sent something!
                #
                # not allowed for recursion problems :-(
                # (maximum recursion depth)
                # self.send_me_last_frame_again()
                #
                # other solution:
                self.t = threading.Thread(
                    target=self._send_video_frame)

        self.t.daemon = True
        self.t.start()

    def _send_audio(self):
        start_time = time.time()
        try:
            _, left_audio, right_audio = self.q_audio.get_nowait()
        except IndexError:
            left_audio, right_audio = self.last_audio
            left_audio = right_audio = None
        except queue.Empty:
            left_audio, right_audio = self.last_audio
            left_audio = right_audio = None
        else:
            self.last_audio = (left_audio, right_audio)

        try:
            if left_audio is not None:
                super(TwitchBufferedOutputStream, self
                    ).send_audio(left_audio, right_audio)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            # Don't call this function again and everything should be
            # cleaned up just fine.
            return

        # send the next frame at the appropriate time
        if left_audio is None:
            downstream_time = 0.0001
        else:
            downstream_time = len(left_audio) / AUDIORATE

        FASTER_SLEEP = 1.0
        if self.next_audio_send_time is None:
            self.t = threading.Timer(downstream_time*FASTER_SLEEP,
                                     self._send_audio)
            self.next_audio_send_time = start_time + downstream_time
        else:
            self.next_audio_send_time += downstream_time
            next_event_time = self.next_audio_send_time - start_time
            if next_event_time > 0:
                self.t = threading.Timer(next_event_time*FASTER_SLEEP,
                                         self._send_audio)
            else:
                # we should already have sent something!
                #
                # not allowed for recursion problems :-(
                # (maximum recursion depth)
                # self.send_me_last_frame_again()
                #
                # other solution:
                self.t = threading.Thread(
                    target=self._send_audio)

        self.t.daemon = True
        self.t.start()

    def send_video_frame(self, frame, frame_counter=None):
        """send frame of shape (height, width, 3)
        with values between 0 and 1

        :param frame: array containing the frame.
        :type frame: numpy array with shape (height, width, 3)
            containing values between 0.0 and 1.0
        :param frame_counter: frame position number within stream.
            Provide this when multi-threading to make sure frames don't
            switch position
        :type frame_counter: int
        """
        if frame_counter is None:
            frame_counter = self.frame_counter
            self.frame_counter += 1

        self.q_video.put((frame_counter, frame))

    def send_audio(self,
                   left_channel,
                   right_channel,
                   frame_counter=None):
        """Add the audio samples to the stream. The left and the right
        channel should have the same shape.

        :param left_channel: array containing the audio signal.
        :type left_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. l can be any integer
        :param right_channel: array containing the audio signal.
        :type right_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. l can be any integer
        :param frame_counter: frame position number within stream.
            Provide this when multi-threading to make sure frames don't
            switch position
        :type frame_counter: int
        """
        if frame_counter is None:
            frame_counter = self.audio_frame_counter
            self.audio_frame_counter += 1

        self.q_audio.put((frame_counter, left_channel, right_channel))

    def get_video_frame_buffer_state(self):
        """Find out how many video frames are left in the buffer.
        The buffer should never run dry, or audio and video will go out
        of sync. Likewise, the more filled the buffer, the higher the
        memory use and the delay between you putting your frame in the
        stream and the frame showing up on Twitch.

        :return integer estimate of the number of video frames left.
        """
        return self.q_video.qsize()

    def get_audio_buffer_state(self):
        """Find out how many audio fragments are left in the buffer.
        The buffer should never run dry, or audio and video will go out
        of sync. Likewise, the more filled the buffer, the higher the
        memory use and the delay between you putting your frame in the
        stream and the frame showing up on Twitch.

        :return integer estimate of the number of audio fragments left.
        """
        return self.q_audio.qsize()

