"""
stream_inference_linux.py

Lightweight Linux-friendly real-time streaming inference for HS-TasNet.
Features:
- List audio devices (PulseAudio/PipeWire via soundcard)
- Live separation from input device and optional playback to output device
- Uses model.init_stateful_transform_fn for low-latency stateful inference

Usage examples:
  python stream_inference_linux.py --list
  python stream_inference_linux.py --model-path ./checkpoint.pt --input-id 2 --output-id 3 --sources 2
  python stream_inference_linux.py --no-audio-output --input-id 2 --sources 2

Keep dependencies minimal: soundcard, torch, numpy
"""

import argparse
import sys
import time
import numpy as np
import torch

from hs_tasnet import HSTasNet

try:
    import soundcard as sc
except Exception:
    print("Error: 'soundcard' not available. Install with: pip install soundcard")
    raise


def list_devices():
    speakers = sc.all_speakers()
    microphones = sc.all_microphones(include_loopback=True)
    default_speaker = sc.default_speaker()
    default_microphone = sc.default_microphone()

    def display_width(text):
        width = 0
        for ch in text:
            if ord(ch) < 128:
                width += 1
            else:
                width += 2
        return width

    def pad(text, width):
        text = text[:width]
        w = display_width(text)
        if w >= width:
            return text
        return text + (' ' * (width - w))

    print("\nSpeakers (output devices):")
    print(f"{'ID':<5} {'Name':<60} {'Ch':<4} {'Default'}")
    print('-' * 80)
    for i, spk in enumerate(speakers):
        name = pad(spk.name, 60)
        ch = getattr(spk, 'channels', 0)
        is_def = ' (default)' if default_speaker and spk.name == default_speaker.name else ''
        print(f"{i:<5} {name} {ch:<4} {is_def}")

    print("\nMicrophones (input devices, include loopback):")
    print(f"{'ID':<5} {'Name':<60} {'Ch':<4} {'Loop'} {'Default'}")
    print('-' * 80)
    for i, mic in enumerate(microphones):
        name = pad(mic.name, 60)
        ch = getattr(mic, 'channels', 0)
        is_loop = 'yes' if getattr(mic, 'is_loopback', False) else 'no'
        is_def = ' (default)' if default_microphone and mic.name == default_microphone.name else ''
        print(f"{i:<5} {name} {ch:<4} {is_loop:<4} {is_def}")


def _device_summary(device, label, default_device):
    if device is None:
        print(f"{label}: None")
        return
    name = getattr(device, 'name', '')
    ch = getattr(device, 'channels', 0)
    is_default = ''
    if default_device and name == getattr(default_device, 'name', None):
        is_default = ' (default)'
    print(f"{label}: {name} (ch={ch}){is_default}")


def build_transform(model: HSTasNet, sources, device):
    # returns a callable that accepts numpy array (channels, frames) and returns (channels, frames)
    return model.init_stateful_transform_fn(return_reduced_sources=sources, device=device, auto_convert_to_stereo=True)


def main():
    parser = argparse.ArgumentParser(description='HS-TasNet streaming inference (Linux)')
    parser.add_argument('--list', action='store_true', help='List audio devices and exit')
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model checkpoint (optional)')
    parser.add_argument('--input-id', type=int, help='Input device ID (from --list)',default=1)
    parser.add_argument('--output-id', type=int, help='Output device ID (from --list)',default=2)
    parser.add_argument('--sources', type=str, default='0', help='Comma separated source indices to keep (e.g. 2 for vocals)')
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
    speakers = sc.all_speakers()
    microphones = sc.all_microphones(include_loopback=True)
    default_speaker = sc.default_speaker()
    default_microphone = sc.default_microphone()

    if args.input_id is not None:
        if args.input_id < 0 or args.input_id >= len(microphones):
            print('Invalid --input-id. Use --list to see valid microphone IDs.')
            return
        mic = microphones[args.input_id]
    else:
        mic = default_microphone

    speaker = None
    if not args.no_audio_output:
        if args.output_id is not None:
            if args.output_id < 0 or args.output_id >= len(speakers):
                print('Invalid --output-id. Use --list to see valid speaker IDs.')
                return
            speaker = speakers[args.output_id]
        else:
            speaker = default_speaker

    in_channels = model.audio_channels
    out_channels = model.audio_channels if not args.no_audio_output else 0

    mic_channels = getattr(mic, 'channels', in_channels) if mic else in_channels
    in_channels = min(in_channels, mic_channels) if mic_channels else in_channels
    if speaker:
        spk_channels = getattr(speaker, 'channels', out_channels)
        out_channels = min(out_channels, spk_channels) if spk_channels else out_channels

    print('Current audio devices:')
    _device_summary(mic, 'Input', default_microphone)
    _device_summary(speaker, 'Output', default_speaker)

    try:
        if mic is None:
            print('No input device available.')
            return

        if args.no_audio_output:
            print('Opening input-only stream...')
            with mic.recorder(samplerate=samplerate,
                              channels=in_channels,
                              blocksize=blocksize) as rec:
                print('Streaming (input-only). Ctrl+C to stop.')
                start = time.time()
                while True:
                    if args.duration and (time.time() - start) >= args.duration:
                        break
                    indata = rec.record(blocksize)
                    audio_chunk = indata.T
                    _ = transform_fn(audio_chunk)
        else:
            if speaker is None:
                print('No output device available. Use --no-audio-output to run input-only.')
                return
            print(f'Opening full-duplex stream @ {samplerate} Hz, block size {blocksize}')
            with mic.recorder(samplerate=samplerate,
                              channels=in_channels,
                              blocksize=blocksize) as rec, \
                 speaker.player(samplerate=samplerate,
                                channels=out_channels,
                                blocksize=blocksize) as play:
                print('Streaming (input->processed output). Ctrl+C to stop.')
                start = time.time()
                while True:
                    if args.duration and (time.time() - start) >= args.duration:
                        break
                    indata = rec.record(blocksize)
                    audio_chunk = indata.T
                    separated = transform_fn(audio_chunk)

                    if hasattr(separated, 'cpu'):
                        separated = separated.cpu().numpy()

                    if separated.ndim == 1:
                        separated = np.expand_dims(separated, 0)

                    out_audio = separated.T

                    if out_audio.shape[1] != out_channels:
                        if out_audio.shape[1] == 1 and out_channels == 2:
                            out_audio = np.repeat(out_audio, 2, axis=1)
                        else:
                            tmp = np.zeros((out_audio.shape[0], out_channels), dtype=out_audio.dtype)
                            minc = min(tmp.shape[1], out_audio.shape[1])
                            tmp[:, :minc] = out_audio[:, :minc]
                            out_audio = tmp

                    play.play(out_audio)
    except KeyboardInterrupt:
        print('\nStopped by user')
    except Exception as e:
        print('\nStream error:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
