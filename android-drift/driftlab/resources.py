"""Process lifetime high-water RSS, including offline data, where OS supports it."""
import os
import sys


def peak_rss():
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                    ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                     'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
            counters=Counters(); counters.cb=ctypes.sizeof(counters)
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL('psapi',use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                raise OSError(ctypes.get_last_error())
            value=int(counters.PeakWorkingSetSize)
        else:
            import resource
            value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value=int(value if sys.platform=='darwin' else value*1024)
        return dict(bytes=value,scope='Process lifetime peak RSS; may include earlier runs',gpu_used=False)
    except (ImportError,OSError,AttributeError) as error:
        return dict(bytes=None,reason=str(error),gpu_used=False)
