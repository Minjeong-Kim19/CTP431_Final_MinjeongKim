# 1. The main idea

This is a system that utilizes a CNN to recognize emotions from facial expressions and adjusts the key, tempo, and various effects of a single piece of music in real time.

# 2. Inspiration

- My Hobby
> Playing background music(BGM) from dramas on my master keyboard.
> ![Hobby](https://raw.githubusercontent.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression/main/Hobby.png)

- BackGround Music(BGM)
> BGM plays an important role in dramas as it changes according to the mood of the work and the emotions of the characters in a particular scene.

- Two types of BGM melodies (The definition provided by me.)

| Different melodies  | The same melody with variations|
|:--------------------|:---------------------|
|entirely different musical progressions|a single melody whose pitch, tempo, or timbre has been altered      => cohesiveness ↑|

- Example of the latter type of BGM from Drama
  - New Life Begin (경경일상, 卿卿日常) (2022)
 
  - ![경경일상 poster](https://raw.githubusercontent.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression/main/경경일상.png)
 
  - Comparison between two scenes 
(Two scenes use same bgm melody with variations.)

|               | The First Scene     | The Second Scene    |
|:------------- |:--------------------|:--------------------|
| Context       | beginning to develop feelings for each other | the matured emotions of the two characters. |
| Starting Note | F note | E note |
| Tempo         | Fast   | Slow   |
| Mood          | Bright | More weight |
| Youtube Link  |[From 9s~](https://www.youtube.com/watch?v=Zz_OWnG5CA0&list=PL2k3iGlQIRUhCF2WtD_70dZcb4MtvuwJ7&index=34)|[From 47s~](https://www.youtube.com/watch?v=OrBGCYg2150&list=PL2k3iGlQIRUhCF2WtD_70dZcb4MtvuwJ7&index=113)| 

# 3. Technical details

 1) Reference from the [GitHub repository](https://github.com/petercunha/Emotion?tab=readme-ov-file) for code

 > The code recognizes human faces and their corresponding emotions from a video or webcam feed.

 > MIT License
 > Copyright (c) 2017 Peter Cunha

- CNN (Convolutional Neural Network) 

  - Conv2D layers : extracting various features from the input facial images. 
  - Softmax activation function : converting the emotion classification results into probabilities.
  - Others(Batch normalization, separable convolutions, max pooling, and global average pooling) : Downsampling and classifying these extracted features. 
  - ![The CNN Architecture used here](https://raw.githubusercontent.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression/main/CNN.png)
  - The image from the research paper [Real-time CNN for Emotion and Gender Classification](https://arxiv.org/abs/1710.07557)

- Haar cascade : detecting face
  - ![Haar cascade](https://raw.githubusercontent.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression/main/Haar.png)
  - a machine learning-based object detection method used to identify faces in images or video streams.
  - It uses a series of classifiers to detect features like edges and textures, making it efficient for real-time face detection tasks.


```python
// Python code with syntax highlighting.
# Webcam and face detection setup
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
```

 2) Musical Effect

- the key, tempo, filters, reverb, wah-wah, and other effects

- Tools (Librosa -> soxr)
  - I improve audio quality by utilizing high-quality resamplers, soxr instead of librosa. 

- First try : Librosa 

```python
// Python code with syntax highlighting.
def process_audio(data, n_steps, playback_speed, effect_type, filter_type):
    audio = np.frombuffer(data, dtype=np.float32)
    
    # Adjust n_fft dynamically based on audio length
    n_fft = min(2048, len(audio))
    
    # Apply pitch shift
    pitched_audio = librosa.effects.pitch_shift(audio, sr=RATE, n_steps=n_steps)
    
    # Apply time stretch
    stretched_audio = librosa.effects.time_stretch(pitched_audio, rate=playback_speed)
    
    # Apply Filter based on filter_type
    if filter_type == "Low-pass":
        stretched_audio = librosa.effects.preemphasis(stretched_audio)
    elif filter_type == "High-pass":
        stretched_audio = librosa.effects.harmonic(stretched_audio)
    
    # Avoid sound distortion by limiting effect intensity
    stretched_audio = np.clip(stretched_audio, -1.0, 1.0)
    
    # Effect Application (Reverb, Delay, Wah-Wah)
    if effect_type == "Reverb":
        stretched_audio = librosa.effects.preemphasis(stretched_audio)
    elif effect_type == "Delay":
        stretched_audio = librosa.effects.harmonic(stretched_audio)
    elif effect_type == "Wah-Wah":
        stretched_audio = librosa.effects.pitch_shift(stretched_audio, sr=RATE, n_steps=-2)
    
    return np.resize(stretched_audio, (CHUNK,))

```

- Second try : soxr

```python
// Python code with syntax highlighting.
def process_audio(data, n_steps, playback_speed, effect_type, filter_type):
    audio = np.frombuffer(data, dtype=np.float32)
    
    # Adjust pitch using soxr's resample
    pitch_factor = 2 ** (n_steps / 12)  # Convert semitone steps to frequency factor
    new_rate = int(RATE * pitch_factor)
    pitched_audio = soxr.resample(audio, RATE, new_rate)
    
    # Adjust tempo using soxr
    tempo_rate = int(new_rate / playback_speed)
    stretched_audio = soxr.resample(pitched_audio, new_rate, tempo_rate)
    
    # Clip the audio to avoid overflow
    processed_audio = np.clip(stretched_audio, -1.0, 1.0)

    # Resize to CHUNK size for playback
    return np.resize(processed_audio, (CHUNK,))
```


 3) Relation between Emotion and Musical mood

 - ![Emotional Responses to Music](https://link.springer.com/article/10.1007/s11031-005-4414-0)
   - By referring to the following research papers, I organized the musical moods corresponding to each emotion and applied effects as follows.
   - ![Graph from the research paper](https://raw.githubusercontent.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression/main/Emotions_Musical_Mood.png)

  | Emotion  | n_steps (Pitch Shift) | Playback Speed (Tempo) | Effect Type | Filter Type |
  |:-------------|:----------------|:----------------------|:-------------|:-------------|
  | Happy |  4(Major) | 2.0(fast) | Reverb | High-pass |
  | Sad |   | -5(Major) | 0.1(slow) | Low-pass |
  | Angry |  7 | 5.0 | Delay | Any | 
  | Surprise | 5 | 4.0 | Reverb | High-pass |
  | Neutral | 0 | 1.0 | None | Low-pass |

  - Applying the musical effects to each emotion

```python   
//Python code with syntax highlighting.
 if current_emotion == 'happy':
     n_steps, playback_speed, effect_type, filter_type = 4, 2.5, "Reverb", "High-pass"  
 elif current_emotion == 'sad':
     n_steps, playback_speed, effect_type, filter_type = -5, 0.3, "Wah-Wah", "Low-pass"  
 elif current_emotion == 'angry':
     n_steps, playback_speed, effect_type, filter_type = 7, 4.0, "Delay", "Any"
 elif current_emotion == 'surprise':
     n_steps, playback_speed, effect_type, filter_type = 5, 2.5, "Reverb", "High-pass"
 else:  # neutral
     n_steps, playback_speed, effect_type, filter_type = 0, 1.0, "None", "Low-pass"
```


# 4. Show the Demo 

>The melody is created by SUNO AI.

>There are three version of the codes.

> **I strongly recommend you to run the code, "emotions_soxr.py" . **

>If you find it difficult to make facial expressions, use the images in the 'facial expression' folder.

> You need to download ffmpeg-7.1.tar.


 1) Run emotions_librosa.py 

> n_steps(Pitch shifting), playback_speed(Tempo), effect_type(Reverb, Delay, WahWah), filter_type(High-pass, Low-pass)

> By Librosa

- Install

```
pip install opencv-python opencv-python-headless numpy keras tensorflow librosa pipwin
pipwin install pyaudio
```

- Run 

```
python emotions_librosa.py
```


 2) Run emotions_soxr.py 

> n_steps(Pitch shifting), playback_speed(Tempo)

> By soxr
 
>[Visit my GitHub Repository : Final_김민정.zip ](https://github.com/Minjeong-Kim19/Dynamic-Music-Mood-Adaptation-Based-on-Facial-Expression)

- Install

```
pip install opencv-python opencv-python-headless numpy keras tensorflow librosa scipy soxr pipwin
pipwin install pyaudio
```

- Run

```
python emotions_soxr.py                                                  
```

 3) Run emotions_soxr.py

> n_steps(Pitch shifting), playback_speed(Tempo), effect_type(Reverb, Delay, WahWah), filter_type(High-pass, Low-pass)

> By soxr

- Install

```
pip install opencv-python opencv-python-headless numpy keras tensorflow librosa scipy soxr pipwin
pipwin install pyaudio
```

- Run

```
python emotions_soxr2.py
```
