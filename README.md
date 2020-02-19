# sippyart

## index

    - wav_to_spectogram.ipynb: given a wav file, show waveform and spectogram
    - app.py: flask app for viewing spectrograms and waveforms of a directory of wav files

    - vaegen.py: given a wav file, trains a vae to reproduce the song for given lengths
    - gan.py: (not working really) trying to generate raw audio signal

## gen usage

    1. first clone repo
    2. move wav file into learning/audio/
    3. train vae on your audio sample by changing FILE_NAMES in `vaegen.py` to your file

## real todo/more feasible

    * change dataloaders for different subset selections (window striding)
    * option to have (len - window_len) # of windows, or (len // window_len)
    * given list of files, create a playlist where the same model trains on all files and creates one wav
    * fix gan

## ideas/misc

    * preprocessing network to determine bpm and scale window size as a factor of beats/measures.
    * style classification
    * key classification
    * neural tuner
    * use nlp model to describe music

## [listen to some of the outputs i got](https://anonstandardunitofmeasurement.bandcamp.com/album/vae)