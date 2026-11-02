"""Fresh-file libdeflate plus private SIMD PNG decoding, RGB8 only.

Unsupported formats fail explicitly in the specialized path. The generic path
uses existing libspng. No decoded images, compressed payloads or answers cached.
"""
import ctypes
import io
import struct
import subprocess
import zlib
import os
import fcntl
from pathlib import Path
import numpy as np
import cv2
import imagecodecs
from PIL import Image

_native=None


def native():
    global _native
    if _native is None:
        src=Path(__file__).resolve().parent/'png_native.c'
        target=Path(os.environ.get('FASTFORWARD_COMPILE_CACHE', '/tmp/fastforward-accel-'+str(os.getuid())))/'png_native.so'
        target.parent.mkdir(parents=True,exist_ok=True)
        with (target.parent/'png_native.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not target.exists():
                temp=target.with_suffix('.'+str(os.getpid())+'.so')
                subprocess.run(['gcc','-O3','-msse4.1','-mssse3','-fPIC','-shared',str(src),'-o',str(temp)],check=True)
                os.replace(temp,target)
        _native=ctypes.CDLL(str(target))
        _native.reconstruct_rgb.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
        _native.reconstruct_rgb.restype=ctypes.c_int
    return _native


def rgb8(data,simd=True,fast_crc=False):
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    at=8;pieces=[];dimensions=None;ended=False
    while at<len(data):
        assert at+12<=len(data),'truncated PNG chunk'
        length=struct.unpack_from('>I',data,at)[0]
        end=at+8+length
        assert end+4<=len(data),'truncated PNG payload'
        tag=data[at+4:at+8];payload=data[at+8:end]
        crc=struct.unpack_from('>I',data,end)[0]
        actual_crc=(imagecodecs.deflate_crc32(tag+payload) if fast_crc else zlib.crc32(tag+payload))
        assert actual_crc==crc,'PNG CRC mismatch'
        if tag==b'IHDR':
            assert dimensions is None and at==8 and length==13
            w,h,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',payload)
            assert depth==8 and color==2 and compression==filtering==interlace==0,'unsupported PNG mode'
            assert 0<w<=16384 and 0<h<=16384 and w*h<=40_000_000
            dimensions=(w,h)
        elif tag==b'IDAT':pieces.append(payload)
        elif tag==b'IEND':ended=True;break
        elif tag[0]&32==0 and tag not in (b'PLTE',):
            raise ValueError('unsupported critical PNG chunk')
        at=end+4
    assert ended and dimensions is not None and pieces
    w,h=dimensions;n=h*(w*3+1)
    raw=np.empty(n+16,np.uint8)
    decoded=imagecodecs.deflate_decode(b''.join(pieces),raw=False,out=raw[:n])
    assert decoded.nbytes==n,'Decompressed PNG size does not match IHDR'
    output=np.empty(h*w*3+16,np.uint8)
    code=native().reconstruct_rgb(raw.ctypes.data,w,h,output.ctypes.data,int(simd))
    assert code==0,('PNG row reconstruction',code)
    return output[:h*w*3].reshape(h,w,3)


def decode(path,side,method):
    with open(path,'rb') as f:data=f.read()
    if method=='private_auto':
        if data.startswith(b'\xff\xd8'):method='private_cv_jpeg8'
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):method='private_deflate_simd_zero'
        else:raise ValueError('Expected JPEG or non-interlaced RGB8 PNG')
    if method=='private_deflate_simd':
        rgb=rgb8(data,True)
    elif method=='private_deflate_simd_zero':
        rgb=rgb8(data,2,True)
    elif method=='private_deflate_scalar':
        rgb=rgb8(data,False)
    elif method=='private_spng_imagecodecs':
        rgb=imagecodecs.spng_decode(data)
    elif method=='private_cv_jpeg8':
        bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_REDUCED_COLOR_8)
        assert bgr is not None
        return cv2.cvtColor(cv2.resize(bgr,(side,side),interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB)
    elif method=='private_jpeg8':
        with Image.open(io.BytesIO(data)) as header:shape=((header.height+7)//8,(header.width+7)//8)
        rgb=imagecodecs.jpeg8_decode(data,shape=shape,outcolorspace='RGB')
        return cv2.resize(rgb,(side,side),interpolation=cv2.INTER_NEAREST)
    elif method=='private_jpeg_draft':
        with Image.open(io.BytesIO(data)) as image:
            image.draft('RGB',(side,side));image.load()
            return np.asarray(image.resize((side,side),Image.Resampling.NEAREST).convert('RGB')).copy()
    else:raise ValueError(method)
    assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
    h,w=rgb.shape[:2]
    reduced=cv2.resize(rgb,(w//4,h//4),interpolation=cv2.INTER_LINEAR_EXACT)
    return cv2.resize(reduced,(side,side),interpolation=cv2.INTER_LINEAR)
