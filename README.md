# HPD-Inference-Audio ReadMe

This repository contains the processing and inferencing code for audio files used in the HPDmobile project. `shift_predictions` is the most up-to-date branch. Need to **merge this with master**. 

Helper files used: 
- gen_argparse.py
- my_functions.py

Author: Maggie Jacoby

## To-Do
- **Add in unpickling file.**
- Check and possibly edit `Process_count_Audio.py` to fit latest practices.
- ~~Modify `4_RF_audio_occ_pred.py` to use `gen_argparse` functions.~~
- ~~Work on plotting code for wave files.~~
- Write code to automatically subset files based on amplitude (similar to or to be used after `copy_audio.py`).
- ~~Write code to split 20-second long files into two 10-second long files.~~
- Split out `process_wav` function from `Process_count_Audio.py`. 


## Process-Audio
Audio was collected in 10-second long .wav files for most homes. A few early homes had 20-second long files. Audio was pickled in some cases, but for most homes raw wav files were transferred directly. 

- Process_count_Audio.py

    This code takes raw wav files and processes them, outputting downsampled and/or dicrete cosine transformed data in arrays. The arrays are saved as .npz files on an hourly basis. Downsampled files are intended to be used to for the public database (`*_ds.npz`) and processed (DCT) files are used in the inferencing algorithms (`*_ps.npz`).

- AudioFunctions.py

    Contains helper functions for `Process_count_Audio.py`.

- copy_audio.py

    This code is similar to `copy_img.py`. It takes in the IMAGE inferences (the 10-second frequency final CSVs, eg stored in `.../H6-black/Inference_DB/BS2/img_inf/` ) and copies the actual audio files that occur in occupied image blocks into labeled folders (eg `.../H6-black/Auto_Labeled/audio_BS2/`).

- audio_to_audio_copy.py

    This code takes files that have been separated and labeled (manually or automatically from `copy_audio.py`), and copies the same timestamp audio wave for a different hub into a labeled folder. 

- process_wav.py

    Function for processing and downsampling audio files
    To be used in Process_count_Audio.py - not implemented yet

- split_way.py

    Takes 20 second wav files and splits into two 10 seconds files (appropriately timestamped).


- Audio_Plotting.ipynb 

    This contains starter code for plotting wave files. **Needs much work**


## Inference-Audio
- audio_confidence.py
    This takes in a path to the processed audio files and outputs occupancy probabilities and decisions (0/1 based on a 50% probability cutoff).


All the model files are located under `Audio_CNN`. 
- `CNN_testing_code`
    This contains labeled audio (under `processed`) for testing the classifier with `load_data.py` (and `load_saved_models_maggieEdits.py`).



## ARCHIVE
- 4_RF_audio_occ_pred.py

    This takes in a path to the processed audio files and outputs occupancy decisions. 

- trained_RL(3.7.6-64).joblib

    This is the old audio classifier...



## Steps for generating inferences on Audio:
0. Unpickle audio files if necessary. Should be in .wav format. 

~~1. Process using `Process_count_Audio.py` to get `*_ps.npz` files. These should be stored in `.../H6-black/BS2/processsed_audio/audio_dct`.~~

1. Process using `Process_count_Audio.py` to get `*_ds.npz` files (downsampled, but not discrete cosine transformed). These should be stored in `.../H6-black/BS2/processsed_audio/audio_downsampled`

2. Perform inference using `audio_confidence.py` on the downsampled files.  Output is a complete daywise csv with 10-second frequency probabilities and predictions stored in `.../H6-black/Inference_DB/BS3/audio_inf/`.