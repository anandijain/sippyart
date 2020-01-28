# audio

index:

    - wav_to_spectogram.ipynb: given a wav file, show waveform and spectogram
    - app.py: flask app for viewing spectrograms and waveforms of a directory of wav files

    - vaegen.py: given a wav file, trains a vae to reproduce the song for given lengths
    - gan.py: (not working really) trying to generate raw audio signal

## gen usage:

* first clone repo
* move wav file into learning/audio/
* train vae

## code snippet

```python
import vaegen
vaegen.train([YOUR WAV FILE], epochs=5, save=True)
```

## [listen](https://anonstandardunitofmeasurement.bandcamp.com/album/vae)



# todo

* other generative models
* preprocessing network to determine bpm and scale window size as a factor of beats/measures.
