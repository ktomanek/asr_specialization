# Copyright 2025 Katrin Tomanek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Run tensorboard in a separate thread to enable proper startup and shutdown from within the server.
from tensorboard import program
import logging
import os
import signal
import socket
import subprocess
import time
from typing import Optional

class TensorBoardManager:
    def __init__(self, logdir: str, port: Optional[int] = None):
        """
        Initialize TensorBoard manager
        Args:
            logdir: Directory containing the logs
            port: Port to run TensorBoard on. If None, will find an available port
        """
        self.logdir = logdir
        self.port = port or self._find_free_port()
        self.process: Optional[subprocess.Popen] = None

    def get_url(self) -> str:
        return 'http://localhost:' + str(self.port) + '/'

    def _find_free_port(self) -> int:
        """Find a free port to run TensorBoard on"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start(self) -> None:
        """Start TensorBoard server"""
        if self.process is not None:
            logging.info("TensorBoard is already running")
            return

        cmd = [
            "tensorboard",
            "--logdir", self.logdir,
            "--port", str(self.port),
            "--bind_all"  # Makes it accessible from other machines
        ]
        
        # Start TensorBoard as a subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it a moment to start up
        time.sleep(3)
        
        if self.process.poll() is not None:
            # Process has terminated
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"TensorBoard failed to start: {stderr.decode()}")
        
        logging.info(f"TensorBoard running on port {self.port}")

    def shutdown(self) -> None:
        """Shutdown TensorBoard server"""
        if self.process is None:
            logging.info("TensorBoard is not running")
            return

        # Try graceful shutdown first
        if os.name == 'nt':  # Windows
            self.process.terminate()
        else:  # Unix-like
            self.process.send_signal(signal.SIGTERM)
            
        # Give it a moment to shut down gracefully
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it didn't shut down gracefully
            self.process.kill()
            self.process.wait()

        self.process = None
        logging.info("TensorBoard server has been shut down")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()