"""
**Debug message printing module**
This module implements a wrapper for printing debug messages.
The wrapper also handles buffering and flushing.
"""

import sys

__all__ = ['MessagePrinter', 'FileMessagePrinter', 'MemoryMessagePrinter', 'stdoutPrinter']


class MessagePrinter(object):
    """
    Message printer base class.
    """
    def __init__(self):
        pass

    def write(self, msg):
        """
        Write a message.
        """
        pass

    def flush(self):
        """
        Flush all buffered messages
        """
        pass


class FileMessagePrinter(MessagePrinter):
    """
    Message printer for writing to a file.
    """
    def __init__(self, file):
        self.file = file

    def write(self, msg):
        self.file.write(msg)

    def flush(self):
        self.file.flush()


class MemoryMessagePrinter(MessagePrinter):
    """
    Message printer for writing to memory.
    """
    def __init__(self):
        self.txt = ""

    def reset(self):
        self.txt = ""

    def messages(self):
        return self.txt

    def write(self, msg):
        self.txt += msg

    def flush(self):
        pass


stdoutPrinter = FileMessagePrinter(sys.stdout)
"""
Message printer object for writing to stdout.
"""
