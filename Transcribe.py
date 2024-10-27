import subprocess
import os
import sys
import json
import time
from transformers import pipeline
import argparse
import re
from GlobalPersonTracker import counter
from GlobalSpeechDiarization import speakercounter

# Initialize the Whisper model globally
whisper_model = None

def audioinference(audio_file, device='cuda:0'):
    """
    Transcribes the given audio file using the Whisper model.

    Args:
        audio_file (str): Path to the audio file.
        device (str): Device to run the model on ('cpu', 'cuda', or 'cuda:0').

    Returns:
        str: Transcribed text from the audio file.
    """
    global whisper_model
    if whisper_model is None:
        # Map device string to device index
        if device == 'cpu':
            device_index = -1  # CPU
        elif device.startswith('cuda'):
            # Extract device index, default to 0
            parts = device.split(':')
            if len(parts) > 1:
                device_index = int(parts[1])
            else:
                device_index = 0
        else:
            device_index = 0  # Default to device 0

        # Initialize the model
        whisper_model = pipeline(
            'automatic-speech-recognition',
            model='openai/whisper-tiny',
            device=device_index
        )
    result = whisper_model(audio_file)
    return result['text']

def extract_chunk(video_path, start_time, duration, output_video, output_audio):
    """Extract a video chunk and audio chunk."""
    try:
        # Extract video chunk
        cmd_video = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "copy",
            "-an",  # Disable audio
            output_video
        ]
        # print(f"\nExecuting video extraction command:\n{' '.join(cmd_video)}")
        result_video = subprocess.run(cmd_video, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result_video.returncode != 0:
            print(f"Error extracting video chunk starting at {start_time}")
            return False

        # Extract audio chunk
        cmd_audio = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-q:a", "0",
            "-vn",  # Disable video
            output_audio
        ]
        # print(f"\nExecuting audio extraction command:\n{' '.join(cmd_audio)}")
        result_audio = subprocess.run(cmd_audio, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result_audio.returncode != 0:
            print(f"Error extracting audio chunk starting at {start_time}")
            return False

        return True
    except Exception as e:
        print(f"Exception during extraction of chunk starting at {start_time}: {e}")
        return False
    
def get_video_duration(video_path):
    """Get the duration of the video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 0
    return duration

def extract_chunks(video_path, device='cuda:0'):
    """Extract 10-second chunks from the video and perform inference on each chunk."""
    # Get the video length
    video_length = get_video_duration(video_path)

    # Determine the number of full 10-second chunks
    num_chunks = int(video_length // 10)

    # Discard last chunk if it's less than 10 seconds
    total_duration = num_chunks * 10
    if total_duration > video_length:
        num_chunks -= 1
        total_duration = num_chunks * 10

    print(f"Total video duration: {video_length:.2f} seconds.")
    print(f"Number of 10-second chunks to process: {num_chunks}.")

    # Create a list to hold the chunks
    chunks = []

    for chunk_number in range(num_chunks):
        start_time = chunk_number * 10
        end_time = start_time + 10

        # Define output paths for video and audio
        output_video = f"chunk_{chunk_number+1}_video.mp4"
        output_audio = f"chunk_{chunk_number+1}_audio.mp3"

        # Append the chunk details to the list
        chunk_info = {
            'chunk_start': start_time,
            'chunk_end': end_time,
            'video': output_video,
            'audio': output_audio
        }
        chunks.append(chunk_info)

        # Extract chunk
        print(f"\nExtracting chunk {chunk_number+1}/{num_chunks} from {start_time:.2f} to {end_time:.2f} seconds.")
        success = extract_chunk(video_path, start_time, 10, output_video, output_audio)
        if not success:
            chunk_info['video_transcribed'] = ""
            chunk_info['audio_transcribed'] = ""
            continue

        # Perform inference on the extracted video chunk
        personcounter = counter(output_video)
        chunk_info['chunk_person_count'] = personcounter['person_tracker_count']
        chunk_info['chunk_global_count'] = personcounter['global_tracker_count']

        # Perform inference on the extracted audio chunk
        audio_transcribed_text = audioinference(output_audio, device=device)
        chunk_info['audio_transcribed'] = audio_transcribed_text
        
        # Get the number of speakers from the speakercounter function (parsing the JSON)
        speaker_counter_json = speakercounter(output_audio)
        speaker_counter_data = json.loads(speaker_counter_json)  # Parse the JSON string
        chunk_info['number_of_speaker'] = speaker_counter_data["count"] # Extract the "count" field
        chunk_info['global_count'] = speaker_counter_data["global_count"]  # Extract the "global count" field
        global_count = speaker_counter_data["global_count"]  # Extract the "global count" field

        # Updated checker logic:
        Expected_count = 2
        if chunk_info['chunk_person_count'] > Expected_count or chunk_info['number_of_speaker'] > Expected_count:
            if chunk_info['chunk_global_count'] > Expected_count or chunk_info['global_count'] > Expected_count:
                chunk_info['check'] = -1
        else:
            chunk_info['check'] = 0

        print(json.dumps(chunk_info, indent=3))
        chunk_number += 1

    # After processing all chunks, delete the chunk files
    # for chunk in chunks:
    #     if os.path.exists(chunk['video']):
    #         os.remove(chunk['video'])
    #     if os.path.exists(chunk['audio']):
    #         os.remove(chunk['audio'])
    # print("Deleted all chunk files.")

    return chunks

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transcribe video into text using both audio and video.')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu")')
    args = parser.parse_args()
    
    video_path = args.input
    output_json_path = args.output
    device = args.device

    if not os.path.isfile(video_path):
        print(f"Error: The input video file '{video_path}' does not exist.")
        sys.exit(1)

    start_time_total = time.time()
    chunks = extract_chunks(video_path, device=device)
    end_time_total = time.time()
    total_elapsed_time = end_time_total - start_time_total

    # Print the extracted chunks info
    print("\nExtracted Chunks:")
    print(json.dumps(chunks, indent=3))
    print(f"\nTotal extraction time: {total_elapsed_time:.2f} seconds.")
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Saved the output to '{output_json_path}'.")