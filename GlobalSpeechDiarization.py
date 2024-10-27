import json
import torchaudio
from pyannote.audio import Pipeline, Inference
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio.transforms as transforms

# Global set of known speaker embeddings and global counter
known_speakers = []
global_speaker_count = 0

# Initialize the diarization pipeline and speaker embedding pipeline
def initialize_pipelines(auth_token):
    # Initialize diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    
    # Initialize speaker embedding pipeline
    embedding_pipeline = Inference("pyannote/embedding", use_auth_token=auth_token)
    
    return diarization_pipeline, embedding_pipeline

def extract_speaker_embeddings(diarization_results, audio_file, embedding_pipeline):
    """
    Extract speaker embeddings for each identified speaker segment.
    Convert the MP3 file into a waveform before processing.
    """
    # Convert MP3 to waveform
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Resample the audio to 16kHz if needed
    resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    
    speaker_embeddings = {}
    
    for segment, _, speaker in diarization_results.itertracks(yield_label=True):
        start_time = int(segment.start * sample_rate)
        end_time = int(segment.end * sample_rate)
        
        # Extract the segment of the waveform corresponding to the speaker
        segment_waveform = waveform[:, start_time:end_time]
        
        # Get the speaker embedding from the embedding pipeline
        embedding = embedding_pipeline({"waveform": segment_waveform, "sample_rate": 16000})
        
        # Convert SlidingWindowFeature to NumPy array and take the mean to get a single embedding
        embedding_numpy = np.mean(embedding.data, axis=0)  # Taking mean over sliding windows

        if speaker not in speaker_embeddings:
            speaker_embeddings[speaker] = []
        speaker_embeddings[speaker].append(embedding_numpy)
    
    # Average embeddings for each speaker
    valid_speaker_embeddings = {}
    for speaker, embeddings in speaker_embeddings.items():
        valid_speaker_embeddings[speaker] = np.mean(embeddings, axis=0)
    
    return valid_speaker_embeddings

def compare_embeddings(new_embedding, known_speakers, threshold=0.25):
    """
    Compare a new speaker embedding with known speaker embeddings using cosine similarity.
    If the similarity is above the threshold, the speaker is considered a known speaker.
    """
    new_embedding_flat = new_embedding.flatten()  # Ensure it's 1D
    for known_embedding in known_speakers:
        known_embedding_flat = known_embedding.flatten()  # Ensure 1D for known embeddings
        similarity = cosine_similarity([new_embedding_flat], [known_embedding_flat])[0][0]
        
        # Debugging: print similarity scores for better understanding
        print(f"Similarity score: {similarity}")
        
        if similarity > threshold:
            return True  # Speaker already known
    return False  # New speaker

def update_global_speakers(new_speaker_embeddings, known_speakers, threshold=0.25):
    """
    Update the global set of known speakers based on new speaker embeddings.
    If a new speaker is detected, add them to the global set and update the global speaker count.
    """
    global global_speaker_count
    current_chunk_speakers = 0  # Track speakers in the current audio chunk

    for speaker, embedding in new_speaker_embeddings.items():
        # Check if the speaker is new
        if not compare_embeddings(embedding, known_speakers, threshold):
            known_speakers.append(embedding)  # Add the new speaker's embedding to the global list
            global_speaker_count += 1  # Increment the global speaker count
        current_chunk_speakers += 1  # Count the speaker in the current chunk

    return current_chunk_speakers

def speakercounter(audio_file, auth_token=""):
    """
    Process an individual audio stream to detect speakers, compare with known speakers,
    and return the count of speakers in the audio snippet and the total number of unique speakers globally.
    
    Args:
        audio_file (str): Path to the audio file.
        auth_token (str): Hugging Face API token for model access.
    
    Returns:
        dict: JSON response containing 'count' (current audio speakers) and 'global_count' (unique speakers globally).
    """
    global known_speakers, global_speaker_count
    
    # Initialize diarization and embedding pipelines (only need to initialize once)
    diarization_pipeline, embedding_pipeline = initialize_pipelines(auth_token)
    
    # Perform diarization on the audio file
    diarization_results = diarization_pipeline(audio_file)
    
    # Extract speaker embeddings from the diarization results
    speaker_embeddings = extract_speaker_embeddings(diarization_results, audio_file, embedding_pipeline)
    
    # Update the global known speakers and count
    current_audio_count = update_global_speakers(speaker_embeddings, known_speakers)
    
    # Return the results as a JSON object
    result = {
        "count": current_audio_count,  # Number of speakers detected in the current audio file
        "global_count": global_speaker_count   # Total number of unique speakers globally
    }
    
    return json.dumps(result)