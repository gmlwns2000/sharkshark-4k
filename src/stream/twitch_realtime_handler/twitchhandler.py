# -*- coding: utf-8 -*-

"""
Parent classes for twitch-realtime-handler
"""

import os
import queue
import subprocess
from dataclasses import dataclass, field
from threading import Thread
from typing import Union

import numpy as np
import streamlink
from streamlink.session import Streamlink


@dataclass
class _TwitchHandler():
    twitch_url: Union[str, None] = None
    chunk_size: int = 2 ** 8
    quality: str = "480p"
    _stream_url: Union[str, None] = None

    def get_stream_url(self) -> None:
        """Retrieve the url of the rtmp stream from twitch url using streamlink"""
        if(os.path.exists(self.twitch_url)):
            print('given path is file', self.twitch_url)
            self._stream_url = self.twitch_url
            return
        
        if self.twitch_url is None:
            raise ValueError("No twitch_url specified")

        try:
            sess = Streamlink()
            stream_hls = sess.streams(self.twitch_url)
            print("TwitchHandler: Found resolutions:", stream_hls.keys())
            if (self.quality not in stream_hls) and self.quality == 'audio_only':
                if "audio_opus" in stream_hls:
                    print("TwitchHandler: opus selected for audio stream")
                    self.quality = 'audio_opus'
                elif "audio" in stream_hls:
                    print("TwitchHandler: audio selected for audio stream")
                    self.quality = 'audio'
                else:
                    self.quality = '360p'
        except streamlink.exceptions.NoPluginError:
            raise ValueError(f"No stream availabe for {self.twitch_url}")
        
        if self.quality not in stream_hls:
            raise ValueError(f"The stream has not the given quality({self.quality}) but ({stream_hls.keys()})")
        #print(stream_hls)
        
        if hasattr(stream_hls[self.quality], 'substreams'):
            print('substream', stream_hls[self.quality].substreams)
            self._stream_url = stream_hls[self.quality].substreams[0].url
        else:
            self._stream_url = stream_hls[self.quality].url


@dataclass
class _TwitchHandlerAudio():
    """Default values for audio"""
    rate: int = 16000  # sampling rate in Hz
    segment_length: float = 2  # length of the audio segment
    quality: str = "audio_only"


@dataclass
class _TwitchHandlerVideo():
    """Default values for video"""
    rate: int = 30  # sampling rate in Hz
    quality: str = "480p"


@dataclass
class _TwitchHandlerGrabber(_TwitchHandler):
    """Parent class for the Audio and Image Grabber"""
    queue_size: int = 1000
    blocking: bool = False
    _th_reader: Union[Thread, None] = field(init=False)
    _n_bytes_per_payload: Union[int, None] = field(init=False)
    _cmd_pipe: Union[list, None] = field(init=False)
    _reshape_size: Union[list, None] = field(init=False)
    dtype: type = field(init=False)
    _terminate: bool = False
    _ffmpeg_thread: Union[Thread, None] = field(init=False)
    _auto_start: bool = True

    def __post_init__(self):
        self._fifo = queue.Queue(maxsize=self.queue_size)

    def terminate(self):
        self._terminate = True
        self._ffmpeg_thread.terminate()

    def _reader(self):
        """Launch the ffmpeg thread and read its output pipe
        and store it into a queue"""
        self._ffmpeg_thread = subprocess.Popen(
            self._cmd_pipe,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            bufsize=10**8
        )
        payload = self._ffmpeg_thread.stdout.read(self._n_bytes_per_payload)
        self._fifo.put(payload)
        while payload:
            if not self.terminate:
                return
            payload = self._ffmpeg_thread.stdout.read(self._n_bytes_per_payload)
            self._fifo.put(payload)

    def _start_thread(self):
        self._th_reader = Thread(target=self._reader, args=(), daemon=True)
        self._th_reader.start()

    def grab(self) -> Union[None, np.array]:
        """Return the image or audio segment"""
        if self._fifo.empty() and not self.blocking:
            return None
        else:
            in_bytes = self._fifo.get()
            return self._bytes_to_array(in_bytes)

    def grab_raw(self) -> bytes:
        if self._fifo.empty() and not self.blocking:
            return None
        else:
            return self._fifo.get()

    def _bytes_to_array(self, in_bytes: bytes) -> np.array:
        """
        Args:
            - in_bytes (bytes): audio segment or frame as bytes

        Returns:
        the frame as a np.array (RGB)
        or a segment as a np.array
        """

        try:
            out = (
                np.frombuffer(in_bytes, self.dtype).reshape(self._reshape_size)
            )
            return out
        except ValueError:
            return None
