import argparse
import sys
import torch
import numpy as np
from hs_tasnet import HSTasNet

try:
    import sounddevice as sd
except ImportError:
    print("Error: 'sounddevice' library not found. Please install it with 'pip install sounddevice'.")
    sys.exit(1)

def list_devices(filter_wasapi=False):
    print("\nAvailable Audio Devices:")
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    
    print(f"{'ID':<5} {'Name':<40} {'API':<15} {'In/Out'}")
    print("-" * 70)
    for i, dev in enumerate(devices):
        api_name = hostapis[dev['hostapi']]['name']
        if filter_wasapi and "WASAPI" not in api_name:
            continue

        in_out = f"{dev['max_input_channels']}/{dev['max_output_channels']}"
        
        # Highlight Loopback devices for WASAPI
        loopback_str = ""
        if api_api := api_name == 'Windows WASAPI' and dev['max_input_channels'] > 0:
            # Check if it likely a loopback device (shared mode usually lists them)
            if "Loopback" in dev['name'] or "再声" in dev['name']: # "再声" is loopback in simplified Chinese Windows
                loopback_str = " [LOOPBACK]"
        
        print(f"{i:<5} {dev['name'][:40]:<40} {api_name:<15} {in_out}{loopback_str}")
    print("\nNote: For Windows system audio capture, look for 'Windows WASAPI' devices with '[LOOPBACK]' or high input channel counts.")

def main():
    parser = argparse.ArgumentParser(description="HS-TasNet Real-time Streaming Inference")
    parser.add_argument("--list", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--wasapi", action="store_true", help="Only list WASAPI devices (Windows only)")
    parser.add_argument("--model-path", type=str, help="Path to pre-trained model checkpoint (optional)")
    parser.add_argument("--input-id", type=int, help="ID of the input device (e.g. WASAPI Loopback)")
    parser.add_argument("--output-id", type=int, help="ID of the output device (e.g. Speaker/Headphones)")
    parser.add_argument("--sources", type=str, default="2", help="Comma separated source indices to keep (0:Drums, 1:Bass, 2:Vocals, 3:Other)")
    
    # We want default to be "small=True". 
    # The standard way to do this with boolean flags is to have a store_false flag (e.g. --large) 
    # OR keep as is but fix the default logic.
    # Current implementation: args.small defaults to True. If user provides --small, it stays True (store_true only sets True).
    # Wait, store_true sets to True if present, default if not.
    # If we want default=True, we should use action="store_false" and name it --large, OR
    # just manually handle it.
    # Let's switch to the standard "negative flag" pattern which is cleaner.
    parser.add_argument("--large", action="store_true", help="Use large model variant (default is small)")
    
    args = parser.parse_args()

    if args.list:
        list_devices(filter_wasapi=args.wasapi)
        return

    # 1. Initialize Model
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = HSTasNet.init_and_load_from(args.model_path)
    else:
        # If user did NOT specify --large, we use small model.
        use_small = not args.large
        variant_str = "small" if use_small else "large"
        print(f"Initializing default {variant_str} model (random weights, for testing)...")
        model = HSTasNet(small=use_small)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model running on: {device}")

    # 2. Parse Sources
    source_indices = [int(s.strip()) for s in args.sources.split(",")]
    print(f"Separating and keeping sources: {source_indices}")

    # 3. Setup Stateful Transform Function
    transform_fn = model.init_stateful_transform_fn(
        return_reduced_sources=source_indices,
        device=device
    )

    # 4. Define Callback
    # We use a single stream if possible, or manual routing if different devices
    def callback(indata, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        
        # indata shape: (frames, channels)
        # We need (channels, frames) for the model
        audio_chunk = indata.T # (2, frames)
        
        # Audio separation
        # Note: transform_fn expects (channels, overlap_len)
        # and returns (channels, overlap_len)
        try:
            separated_chunk = transform_fn(audio_chunk)
            
            # Convert back to (frames, channels) for sounddevice
            # Handle mono/stereo output
            if separated_chunk.ndim == 1:
                # Mono to stereo
                out_audio = np.stack([separated_chunk] * 2, axis=1)
            else:
                out_audio = separated_chunk.T
                
            outdata[:] = out_audio
        except Exception as e:
            print(f"Error in callback: {e}")
            outdata.fill(0)

    # 5. Start Streaming
    samplerate = model.sample_rate
    blocksize = model.overlap_len
    
    print(f"\nStarting stream...")
    print(f"Sample Rate: {samplerate} Hz")
    print(f"Block Size: {blocksize} samples ({blocksize/samplerate*1000:.2f} ms)")

    try:
        with sd.Stream(
            device=(args.input_id, args.output_id),
            samplerate=samplerate,
            blocksize=blocksize,
            dtype='float32',
            channels=(model.audio_channels, 2), # Assuming output is stereo
            callback=callback
        ):
            print("Streaming active. Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as e:
        print(f"\nStream failed: {e}")

if __name__ == "__main__":
    main()
