"""
process_wav.py

Function for processing and downsampling audio files
to used in Process_count_Audio.py - not implemented yet
"""

def process_wav(wav_name, date_folder_path, minute, fs=8000):
    wav_path = os.path.join(date_folder_path, minute, wav_name)
    t = wav_name.split(' ')[-1].strip('._audio.wav')
    time_file = f'{t[0:2]}:{t[2:4]}:{t[4:6]}'

    try:  
        _, wav = scipy.io.wavfile.read(wav_path) # _ = fs
        audio_len_seconds = len(wav)/fs # length of audio clip in seconds
        all_seconds.append(time_file)
        assert (audio_len_seconds == 10.0)
        
        ## Process Audio
        processed_audio = np.zeros((int(len(wav)),number_of_filters)) # Placeholder
        
        temp = butter_lowpass_filter(wav, 100, fs, order=6) # low pass filter (first filter)
        temp -= np.mean(temp) # Mean Shift
        processed_audio[:,0] = abs(temp) # Full wave rectify

        for idx, Filter in enumerate(filter_banks):
            temp = butter_bandpass_filter(wav, Filter[0], Filter[1], fs, order=6) # Band pass filter
            processed_audio[:, idx+1] = abs(temp) # Full wave rectify

        ## Downsample:
        downsampled = np.zeros((num_final_datapoint,number_of_filters))

        for i in range(number_of_filters):
            downsampled[:,i] = processed_audio[filter_i_sampling_index[:,i],i]

        ################ Comment the following lines if don't want to perform dct ################
        processed_audio = dct(downsampled) # Perform DCT across different filter on each timepoint
        processed_audio = processed_audio[:,:12] # Keep only first 12 coefficients
        processed_audio = scale(processed_audio,axis=1) # Normalizing/Scaling to zero mean & unit std      ?? Look into this                 
        ################################################################
  
        return processed_audio, downsampled, time_file
    
    except Exception as e:
        print(f'Error processing file {wav_path}: {e}')
        return [], [], time_file