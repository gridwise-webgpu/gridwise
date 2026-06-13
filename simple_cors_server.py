#!/usr/bin/env python3
# encoding: utf-8
"""Use instead of `python3 -m http.server` when you need CORS"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        return super(CORSRequestHandler, self).end_headers()


class ReuseAddrHTTPServer(HTTPServer):
    allow_reuse_address = True


port = int(os.environ.get("PORT", 8000))
httpd = ReuseAddrHTTPServer(("127.0.0.1", port), CORSRequestHandler)
httpd.serve_forever()
