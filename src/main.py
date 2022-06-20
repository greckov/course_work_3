#!/usr/bin/env python3
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

import recognizer


def _parse_run_arguments():
    parser = ArgumentParser(
        prog='Text Recognizer',
        description='Recognizes text from the provided file path',
    )

    # Add available command arguments
    parser.add_argument('path', type=Path, help='Image file path (only PNG supported)')
    parser.add_argument(
        '--format',
        required=False,
        choices=('text', 'json'),
        default='text',
        help='Specify output data format (text or json)',
    )

    return parser.parse_args()


def _validate_image_type(path):
    supported_formats = ('PNG',)

    try:
        with Image.open(path) as image:
            fmt = image.format

        if fmt not in supported_formats:
            print(f'ERROR: Unsupported image format. Supported are: {supported_formats}', file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f'ERROR: Provided file not found')
        sys.exit(1)


def _invoke_recognition(path):
    model = recognizer.get_trained_model()

    return recognizer.predict_single_object(str(path.resolve()), model)


if __name__ == '__main__':
    args = _parse_run_arguments()
    _validate_image_type(args.path)

    recognition_result = _invoke_recognition(Path(args.path))

    if args.format == 'text':
        print(recognition_result)
    else:
        body = {
           'text': recognition_result,
        }
        print(json.dumps(body))
