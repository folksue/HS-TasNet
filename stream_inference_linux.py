"""
stream_inference_linux.py

Lightweight Linux-friendly real-time streaming inference for HS-TasNet.
Features:
- List audio devices
- Live separation from input device and optional playback to output device
- Uses model.init_stateful_transform_fn for low-latency stateful inference

Usage examples:
  python stream_inference_linux.py --list
  python stream_inference_linux.py --model-path ./checkpoint.pt --input-id 2 --output-id 3 --sources 2
  python stream_inference_linux.py --no-audio-output --input-id 2 --sources 2

Keep dependencies minimal: sounddevice, torch, numpy
"""

import argparse
import sys
import time
import numpy as np
import torch

from hs_tasnet import HSTasNet

try:
    import sounddevice as sd
except Exception:
    print("Error: 'sounddevice' not available. Install with: pip install sounddevice")
    raise


def list_devices():
    devs = sd.query_devices()
    default_input, default_output = sd.default.device
    print("\nAvailable audio devices:")
    print(f"{'ID':<5} {'Name':<45} {'In':<4} {'Out':<4} {'Default'}")
    print('-' * 80)
    for i, d in enumerate(devs):
        name = d['name'][:45]
        in_ch = d['max_input_channels']
        out_ch = d['max_output_channels']
        is_def = ''
        if i == default_input:
            is_def += ' (default-in)'
        if i == default_output:
            is_def += ' (default-out)'
        print(f"{i:<5} {name:<45} {in_ch:<4} {out_ch:<4} {is_def}")


def build_transform(model: HSTasNet, sources, device):
    # returns a callable that accepts numpy array (channels, frames) and returns (channels, frames)
    return model.init_stateful_transform_fn(return_reduced_sources=sources, device=device, auto_convert_to_stereo=True)


def main():
    parser = argparse.ArgumentParser(description='HS-TasNet streaming inference (Linux)')
    parser.add_argument('--list', action='store_true', help='List audio devices and exit')
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model checkpoint (optional)')
    parser.add_argument('--input-id', type=int, help='Input device ID (from --list)')
    parser.add_argument('--output-id', type=int, help='Output device ID (from --list)')
    parser.add_argument('--sources', type=str, default='2', help='Comma separated source indices to keep (e.g. 2 for vocals)')
    parser.add_argument('--large', action='store_true', help='Use large model variant (default: small)')
    parser.add_argument('--no-audio-output', action='store_true', help='Do not play audio to output device (process only)')
    parser.add_argument('--duration', type=float, default=None, help='Duration in seconds to run (default: until Ctrl+C)')

    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    # load / init model
    if args.model_path:
        print(f'Loading model from {args.model_path}...')
        model = HSTasNet.init_and_load_from(args.model_path)
    else:
        model = HSTasNet(small=not args.large)
        print('Initialized default model (random weights).')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    sources = [int(s.strip()) for s in args.sources.split(',') if s.strip() != '']
    if len(sources) == 0:
        print('Provide at least one source index with --sources')
        return

    transform_fn = build_transform(model, sources, device)

    samplerate = int(model.sample_rate)
    blocksize = int(model.overlap_len)
    in_channels = model.audio_channels
    out_channels = model.audio_channels if not args.no_audio_output else 0

    # callback for bi-directional Stream (input+output)
    def stream_callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            # indata shape: (frames, in_ch)
            audio_chunk = indata.T  # (channels, frames)

            separated = transform_fn(audio_chunk)

            if hasattr(separated, 'cpu'):
                separated = separated.cpu().numpy()

            if separated.ndim == 1:
                separated = np.expand_dims(separated, 0)

            out_audio = separated.T  # (frames, channels)

            # match output channels
            chc = outdata.shape[1]
            if out_audio.shape[1] != chc:
                if out_audio.shape[1] == 1 and chc == 2:
                    out_audio = np.repeat(out_audio, 2, axis=1)
                else:
                    tmp = np.zeros_like(outdata)
                    minc = min(tmp.shape[1], out_audio.shape[1])
                    tmp[:, :minc] = out_audio[:, :minc]
                    out_audio = tmp

            outdata[:] = out_audio
        except Exception as e:
            print('Callback error:', e, file=sys.stderr)
            outdata.fill(0)

    # callback for input-only stream
    def input_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            audio_chunk = indata.T
            _ = transform_fn(audio_chunk)
            # no playback; user can extend to save or forward
        except Exception as e:
            print('Input callback error:', e, file=sys.stderr)

    stream_device = None
    if args.input_id is not None or args.output_id is not None:
        stream_device = (args.input_id, args.output_id) if not args.no_audio_output else (args.input_id, None)

    try:
        if args.no_audio_output:
            print('Opening input-only stream...')
            with sd.InputStream(device=args.input_id,
                                channels=in_channels,
                                samplerate=samplerate,
                                blocksize=blocksize,
                                dtype='float32',
                                callback=input_callback):
                print('Streaming (input-only). Ctrl+C to stop.')
                if args.duration:
                    time.sleep(args.duration)
                else:
                    while True:
                        time.sleep(1)
        else:
            # use single Stream for full duplex where supported
            print(f'Opening full-duplex stream @ {samplerate} Hz, block size {blocksize}')
            with sd.Stream(device=stream_device,
                           samplerate=samplerate,
                           blocksize=blocksize,
                           dtype='float32',
                           channels=max(in_channels, out_channels),
                           callback=stream_callback):
                print('Streaming (input->processed output). Ctrl+C to stop.')
                if args.duration:
                    time.sleep(args.duration)
                else:
                    while True:
                        time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopped by user')
    except Exception as e:
        print('\nStream error:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
