import os
import argparse
from art import tprint


def main(args):
    tprint('Oficina - Data Mining')
    os.system("uvicorn app:app --reload --port " + args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='API Oficina')
    parser.add_argument('--port', type=str, dest='port', help='Port ', default='5000')
    main(parser.parse_args())
