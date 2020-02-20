#!/usr/bin/python

__author__ = "Jackson Murrell"
__maintainer__ = "Jackson Murrell"
__version__ = "1.3.3"
__email__ = "jackson.m.murrell@protonmail.com"
__status__ = "Production"
__credits__ = ["Jackson Murrell", "Kevin Mehta", "Duncan McPhie"]

import sys, argparse

import agent
from driver import experiment
from vis import visualize_experiment

def exception_handler(exception_type, exception, traceback, debug_hook=sys.excepthook):
    import pdb
    debug_hook(exception_type, exception, traceback)
    pdb.post_mortem(traceback)

def bp():
    import pdb;pdb.set_trace()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", required=False, action="store_true", default=False,
                        help="Enter a post-mortem debug shell if the program encounters an error. ")

    args = parser.parse_args()

    if args.debug:
        sys.excepthook = exception_handler

    return visualize_experiment()

    return 0


import pygame
bp()
pygame.init()
pygame.display.list_modes()

if __name__ == "__main__":
    sys.exit(main())

