"""Keep an X11 window above the others (EWMH _NET_WM_STATE_ABOVE), stdlib only.

OpenCV's Qt highgui ignores WND_PROP_TOPMOST on Linux (checked 2026-09-29 under
gnome-shell: _NET_WM_STATE stayed FOCUSED only), and wmctrl/xdotool/python-xlib
are not installed on worker1. This sends the same client message wmctrl would,
via ctypes on libX11. Best effort: every failure returns False, never raises.
"""
import ctypes
import ctypes.util
import os
import re
import subprocess


class _XClientMessageEvent(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong),
                ('send_event', ctypes.c_int), ('display', ctypes.c_void_p),
                ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong),
                ('format', ctypes.c_int), ('data', ctypes.c_long * 5)]


class _XEvent(ctypes.Union):
    _fields_ = [('xclient', _XClientMessageEvent), ('pad', ctypes.c_long * 24)]


_CLIENT_MESSAGE = 33
_SUBSTRUCTURE_REDIRECT_MASK = 1 << 20
_SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
_NET_WM_STATE_ADD = 1


def find_window(name):
    """X window id of the top-level window titled `name`, or None."""
    try:
        out = subprocess.run(['xwininfo', '-name', name, '-int'], capture_output=True,
                             text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'Window id:\s+(\d+)', out)
    return int(m.group(1)) if m else None


def raise_above(name):
    """Ask the window manager to keep window `name` above others and activate it."""
    if not os.environ.get('DISPLAY'):
        return False
    wid = find_window(name)
    path = ctypes.util.find_library('X11')
    if wid is None or path is None:
        return False
    try:
        x = ctypes.cdll.LoadLibrary(path)
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XInternAtom.restype = ctypes.c_ulong
        x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x.XDefaultRootWindow.restype = ctypes.c_ulong
        x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                 ctypes.c_long, ctypes.POINTER(_XEvent)]
        x.XFlush.argtypes = [ctypes.c_void_p]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        d = x.XOpenDisplay(None)
        if not d:
            return False
        try:
            root = x.XDefaultRootWindow(d)
            mask = _SUBSTRUCTURE_REDIRECT_MASK | _SUBSTRUCTURE_NOTIFY_MASK
            for mtype, data in (
                    ('_NET_WM_STATE', (_NET_WM_STATE_ADD,
                                       x.XInternAtom(d, b'_NET_WM_STATE_ABOVE', 0), 0, 1, 0)),
                    ('_NET_ACTIVE_WINDOW', (1, 0, 0, 0, 0))):
                ev = _XEvent()
                ev.xclient.type = _CLIENT_MESSAGE
                ev.xclient.send_event = 1
                ev.xclient.window = wid
                ev.xclient.message_type = x.XInternAtom(d, mtype.encode(), 0)
                ev.xclient.format = 32
                for i, v in enumerate(data):
                    ev.xclient.data[i] = v
                x.XSendEvent(d, root, 0, mask, ctypes.byref(ev))
            x.XFlush(d)
        finally:
            x.XCloseDisplay(d)
        return True
    except Exception:
        return False
