"""Cancelable native model HTTP streams without speculative inference deadlines."""
from __future__ import annotations

import http.client
import json
import socket
import threading
from urllib.parse import urlsplit

import requests


class NativeCompletionResponse:
    """Own the socket before waiting for headers, so explicit cancellation works.

    Connection setup and transmission retain a transport deadline. Once the
    request is sent to a verified local backend, inference has no guessed read
    deadline. Its independent activity monitor supplies observability instead.
    """
    def __init__(self, url, payload, *, headers=None, connect_timeout=10):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("native model URL must be HTTP or HTTPS")
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(parsed.hostname, parsed.port, timeout=connect_timeout)
        self.url = url
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.request_headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **dict(headers or {})}
        self.encoding = "utf-8"
        self.status_code = 0
        self.headers = {}
        self.reason = ""
        self.request = None
        self.raw = self
        self.response = None
        self._socket = None
        self.cancelled = threading.Event()
        self._text = None
        self.transport_timeout = connect_timeout

    def open(self):
        self.connection.connect()
        self._socket = self.connection.sock
        if self.cancelled.is_set():
            self.connection.close()
            raise OSError("model request explicitly canceled before dispatch")
        self.connection.request("POST", self.path, body=self.payload, headers=self.request_headers)
        if self.cancelled.is_set():
            self.shutdown()
            raise OSError("model request explicitly canceled during dispatch")
        if self.connection.sock is not None:
            self.connection.sock.settimeout(None)
        self.response = self.connection.getresponse()
        self.status_code = self.response.status
        self.reason = self.response.reason
        self.headers = dict(self.response.getheaders())
        if not 200 <= self.status_code < 300 and self._socket is not None:
            # An error response is transport data, not active inference.
            self._socket.settimeout(self.transport_timeout)

    def shutdown(self):
        self.cancelled.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise requests.HTTPError(f"{self.status_code} {self.reason} for {self.url}", response=self)

    @property
    def text(self):
        if self._text is None:
            self._text = self.response.read().decode(self.encoding, errors="replace") if self.response is not None else ""
        return self._text

    def iter_lines(self, *, decode_unicode=False):
        if self.response is None:
            raise RuntimeError("model stream is not open")
        while True:
            line = self.response.readline()
            if not line:
                return
            yield line.decode(self.encoding) if decode_unicode else line

    def close(self):
        self.shutdown()
        self.connection.close()
        if self.response is not None:
            self.response.close()
