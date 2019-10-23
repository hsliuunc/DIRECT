"""
**A unique identifier generator module**

Also provides the dfo revision number.
This module is imported automatically when dfo is imported.
"""

from os import getpid
from socket import gethostname

__all__ = ['locationID', 'revision']

revision_str = "$Rev: 773 $"

revision = int(revision_str.split(' ')[1])

# Unique location fingerprint for debug output
# Get host ID (IP and hostname), works only for IPv4
# (myName, myAliases, myIPs)=gethostbyname_ex(gethostname())
myName = gethostname()

# Task id
pid = getpid()

tid = 0
"Microthread ID. Set by module :mod:`dfo.parallel.cooperative`"


# Fingerprint: hostname_pid_microthread
def locationID():
    """
    Generates a unique indentifier for every microthread.
    Form: "hostname_pid_microthread"
    """
    return "%s_%x_%x" % (myName, pid, tid)
