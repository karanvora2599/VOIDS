from pyannote.audio import Pipeline

# Initialize the pipeline with authentication token
def speakercounter(audio_file, auth_token="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"):
    # Load the pre-trained speaker diarization pipeline with authentication
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    
    # Apply diarization on the input audio file
    diarization = pipeline(audio_file)
    
    # Create a set to store unique speakers
    unique_speakers = set()

    # Iterate through diarization results and collect speaker labels
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        unique_speakers.add(speaker)

    # Return the number of unique speakers
    return len(unique_speakers)