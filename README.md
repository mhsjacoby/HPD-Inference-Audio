# HPD-Inference-Audio ReadMe

This repository contains the processing and inferencing code for audio files used in the HPDmobile project. `shift_predictions` is the most up-to-date branch. Need to **merge this with master**. 

Helper files used: 
- gen_argparse.py
- my_functions.py

Author: Maggie Jacoby

## To-Do
- **Add in unpickling file.**
- Check and possibly edit `Process_count_Audio.py` to fit latest practices.
- Modify `4_RF_audio_occ_pred.py` to `gen_argparse` functions.
- Work on plotting code for wave files.
- Write code to automatically subset files based on amplitude (similar to or to be used after `copy_audio.py`).
- Write code to split 20-second long files into two 10-second long files.


## Process-Audio
Audio was collected in 10-second long .wav files for most homes. A few early homes had 20-second long files. Audio was pickled in some cases, but for most homes raw wav files were transferred directly. 

- Process_count_Audio.py

    This code takes raw wav files and processes them, outputting downsampled and/or dicrete cosine transformed data in arrays. The arrays are saved as .npz files on an hourly basis. Downsampled files are intended to be used to for the public database (`*_ds.npz`) and processed (DCT) files are used in the inferencing algorithms (`*_ps.npz`).

- AudioFunctions.py

    Contains helper functions for `Process_count_Audio.py`.

- copy_audio.py

    This code is similar to `copy_img.py`. It takes in the IMAGE inferences (the 10-second frequency final CSVs, eg stored in `.../H6-black/Inference_DB/BS2/img_inf/` ) and copies the actual audio files that occur in occupied image blocks into labeled folders (eg `.../H6-black/Auto_Labeled/audio_BS2/`).


## Inference-Audio
- 4_RF_audio_occ_pred.py

    This takes in a path to the processed audio files and outputs occupancy decisions. 

- Audio_Plotting.ipynb 

    This contains starter code for plotting wave files. **Needs much work**

- trained_RL(3.7.6-64).joblib

    This is the audio classifier...



## Steps for generating inferences on Audio:
0. Unpickle audio files if necessary. Should be in .wav format. 

1. Process using `Process_count_Audio.py` to get `*_ps.npz` files. These should be stored in `.../H6-black/BS2/processsed_audio/audio_dct`.

2. Perform inference using `4_RF_audio_occ_pred.py` on the processed (DCT) files.  Output is a complete daywise csv with 10-second frequency predictions stored in `.../H6-black/Inference_DB/BS3/audio_inf/`.