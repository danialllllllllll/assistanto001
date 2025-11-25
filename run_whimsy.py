#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from learning.app_server import start_server

if __name__ == "__main__":
    start_server()
