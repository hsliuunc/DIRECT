"""
**Debug message generation module
"""

from .identify import locationID
from .dbgprint import stdoutPrinter
import sys, time


__all__ = ['DbgSetup', 'DbgDefaultPrint', 'DbgSetDefaultPrinter', 'DbgMsg', 'DbgMsgOut']

prependTime = False
timePrecision = 1
defaultPrinter = stdoutPrinter
prependPrefix = True


def DbgSetDefaultPrinter(p):
    global defaultPrinter
    defaultPrinter = p


def DbgSetup(pTime=False, tPrec=1, prefix=True):
    global prependTime, timePrecision, prependPrefix
    prependTime = pTime
    timePrecision = tPrec
    prependPrefix = prefix


def DbgDefaultPrint(msg):
    defaultPrinter.write(msg)
    defaultPrinter.flush()


# Format a debug message.
# Text can be a multiline text. Prefix every line with locationID and subsystem.
def DbgMsg(subsystem, text):
    """
    Generates a debug message with *text* in its body.
    """
    rows = text.split("\n")

    if not prependPrefix:
        return "\n".join(rows)

    if prependTime:
        t = time.time()
        prefix = "%.*f %s %s: " % (timePrecision, t, locationID(), subsystem)
    else:
        prefix = "%s %s: " % (locationID(), subsystem)

    out = []
    for row in rows:
        out.append(prefix + row)
    return "\n".join(out)


# Format and print debug message.
def DbgMsgOut(subsystem, text, printer=None):
    """
    Generates and prints using the default message printer.
    """
    p = defaultPrinter if printer is None else printer

    rows = text.split("\n")

    if not prependPrefix:
        for row in rows:
            p.write(row + "\n")
    else:
        if prependTime:
            t = time.time()
            prefix = "%.*f %s %s: " % (timePrecision, t, locationID(), subsystem)
        else:
            prefix = "%s %s: " % (locationID(), subsystem)
        for row in rows:
            p.write(prefix + row + "\n")

    p.flush()
