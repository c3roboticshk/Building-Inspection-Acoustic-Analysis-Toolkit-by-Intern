# Building Inspection Acoustic Analysis Toolkit

## Overview
This toolkit analyzes acoustic knocking sounds on building surfaces to 
detect structural properties. It extracts audio features from knocking 
samples, performs clustering to identify material patterns, and projects 
the results onto building blueprints for visualization.

## Key Features
- 🎵 Acoustic feature extraction from WAV files
- 🧠 Machine learning clustering (KMeans, DBSCAN, Spectral)
- 🖼️ Heatmap generation for wall condition visualization
- 🏢 Perspective projection onto building blueprints
- 📊 Feature analysis

## File Structure
project-root/

└── README.md

Analysis
├── attach.py           # Main image projection module

├── folder_test.py      # Folder processing (clustering + visualization)

├── folder_test_with_n.py  # Enhanced folder processing with autoencoder

├── plot.py             # Feature analysis and plotting

├── single_test.py      # Single WAV file analysis

├── test.py             # Multi-wall analysis

├── test_with_n.py      # Multi-wall analysis with autoencoder

└── model.py            # Neural network combined with spectral analysis (data insufficient, still in progress)

Denoise
├──cut_freq.py

└──timedff.py

Annotation
├──drawPerspective.py

└──paint.html           



## Core Modules

### 1. Audio Processing (`single_test.py`, `plot.py`)
- Detects knocking sounds in WAV files
- Extracts acoustic features:
  - Duration, RMS energy, peak amplitude
  - Decay time, spectral centroid
  - Frequency ratios (low/mid/high)
  - Dominant frequency
- Visualizes waveforms and spectra
- Two features stand out in detection purpose: rms energy and middle frequency ratio (500Hz - 1100Hz), other features still need varify
- in analyze_knocks: Decay time is excluded due because data accuracy is difficult to grasp，threshold coefficient can be modified to get proper decay time to separate defect voices
  ```python
  Decay time (time for amplitude to drop to 10% of peak)
        peak_amplitude = np.max(np.abs(knock_segment))
        threshold = 0.2 * peak_amplitude
        above_threshold = np.where(np.abs(knock_segment) > threshold)[0]
        if len(above_threshold) > 0:
            decay_time = (above_threshold[-1] - above_threshold[0]) / sample_rate
        else:
            decay_time = 0
  
- in analyze_knocks: increase height to exclude noise or decrease to include faint knocking
   ```python
  # Find the knocking sounds (peaks in amplitude)
    peaks, _ = find_peaks(np.abs(audio_data), height=0.05, distance=sample_rate * 0.5)

### 2. Clustering Analysis (`folder_test.py`, `folder_test_with_n.py`)
- Processes folders of WAV files
- Performs clustering using:
  - KMeans
  - DBSCAN
  - Spectral Clustering
- Generates:
  - Cluster heatmaps (`plot.png`)
  - Feature scatter plots (`dim.png`)
  - Cluster distribution reports

### 3. Building Projection (`attach.py`)
- Projects wall analysis results onto building blueprint
- Uses perspective transformation
- Handles multiple wall segments with custom coordinates
- Outputs final building image (`projected_building.jpg`)

### 4. Advanced Feature Extraction (`*_with_n.py`)
- Uses autoencoder for dimensionality reduction
- 8-layer neural network architecture
- Latent space feature extraction
- Improved clustering separation

## Usage Workflow

1. **Collect Data**:
   - Record knocking sounds on building surfaces
   - Organize WAV files by wall section (e.g., `/West/west_s1(19)/001.wav--163.wav`)

2. **Analyze Single Wall**:

    in folder_test.py or folder_test_with_n.py:
    in main:
   - Modify 
     ```python
     wav_folder = "/path/to/wall/section"
     
   - Choose cluster method：  
     ```python
     df_all = pd.DataFrame(all_features)
     feature_cols = ['rms_energy','mid_freq_ratio']
     .... 
     labels, best_n = perform_clustering(X_scaled,method='dbscan' or 'KMeans' or 'spectral' or 'gmm')
     
   - Input image of the building:
     ```python
     building_img = cv2.imread('/path/to/building_blueprint.jpg')
     
   - Define projected area: 
     ```python
     building_corners = np.array([[upper left], [upper right], [lower right], [lower left]], dtype=np.float32)
     
   - Results are shown in dim.png (features layout), dex.png (projected wall), plot.png (heatmap result)
   
    ```bash
   python folder_test_with_n.py
   python folder_test.py
   
   
   

3. **Full Building Analysis**: 

    in test.py or test_with_n.py:
    in main:
   - Modify:
     ```python
     wav_folder = "/path/to/wall"
     
   - Choose cluster method：  
     ```python
     df_all = pd.DataFrame(all_features)
     feature_cols = ['rms_energy','mid_freq_ratio']
     .... 
     labels, best_n = perform_clustering(X_scaled,method='dbscan' or 'KMeans' or 'gmm')  'spectral' dose not apply

   - Results are shown in dim.png (features layout), result.png (heatmap result)
   
    ```bash
   python test_with_n.py
   python test.py
   
4. **Project Results (only after Full Building Analysis)**: 
    
    in attach.py:
    
    - modify：  
      ```python
      BASE_PATH = "/path/to/building/sections/"
      BG_PATH = "/path/to/building_blueprint.jpg"
      WALL_COORDINATES = {
      "wall_section1": np.array([[x1,y1], [x2,y2], ...]),
      ...
      }
    - Results are shown in projected_building.jpg
   ```bash
    python attach.py
   
5. **Audio analysis**: 
    
    in single_test.py:
    
    - modify：  
      ```python
       # Replace with your WAV file path
       wav_file_path = (
        '/Users/ywsun/Desktop/pythonProject/C3/audio_samples/audio/no_fan_pc_usbc.wav')
       knock_save_dir = "/Users/ywsun/Desktop/pythonProject/C3/audio_samples/audio/cut/5/"
       # Analyze the WAV file and save knock segments
       results, sample_rate, audio_data, peaks = analyze_knocks(wav_file_path, save_dir=None)
       # Plot and display results
       plot_results(results, sample_rate, audio_data, peaks)
    
   ```bash
    python single_test.py

   
## Dependencies
numpy==1.26.0
opencv-python==4.9.0.80
scipy==1.13.0
pandas==2.2.1
matplotlib==3.8.2
scikit-learn==1.4.0
tensorflow==2.15.0

## Appendix

1. **model.py**

 🔧 Key Components
1. Neural Network Model
   - Binary classifier (good/bad knocks)
   - Architecture: 8-neuron → 4-neuron → output layers with dropout
   - Trained with Adam optimizer and early stopping

2. Signal Processing
   - Computes fixed-length FFT spectra (`compute_fft`)
   - Measures spectral similarity using cosine similarity
   - Extracts audio features: RMS energy, frequency band ratios

3. Feature Engineering
   - Creates transformed features:
     - Squared terms: `low_freq_ratio²`, `high_freq_ratio²`
     - Trigonometric transforms: `sin(rms_energy)`, `sin(mid_freq_ratio)`

4. Hybrid Quality Assessment
   - Combines model predictions (70%) with spectral reward (30%)
   - Spectral reward based on similarity to good/bad reference spectra

---

⚙️ Workflow
1. Training Phase
   - Input: One "good" and one "bad" sample WAV file
   - Generates reference spectra for good/bad knocks
   - Builds NN model and saves:
     - Model weights (`knock_detection_model.h5`)
     - Feature scaler (`knock_scaler.npy`)
     - Reference spectra (`good_ref_spectrum.npy`, `bad_ref_spectrum.npy`)

2. Inference Phase
   - Processes WAV files in target folder
   - For each detected knock:
     - Extracts features
     - Computes prediction probability
     - Calculates spectral reward
     - Determines final quality score
   - Generates:
     - CSV report with per-knock details
     - Cluster heatmap
     - Feature scatter plots

---

📊 Key Outputs
1. Per-file summary:
   - Total knocks detected
   - Good knocks count
   - Quality rate (percentage)
   - Pass/Fail status

2. Visualizations:
   - Model training history (accuracy/loss plots)
   - Cluster heatmap
   - Feature distribution scatter plots

---

⚠️ Dependencies
- Requires `folder_test.py` module (contains `analyze_knocks`, `plot_cluster_heatmap`, `plot_features`)
- Uses TensorFlow/Keras, Scikit-learn, NumPy, Pandas, Matplotlib

---

💻 Execution
Run directly to:
1. Train model using reference audio samples
2. Analyze all WAV files in specified folder
3. Generate diagnostic reports and visualizations

Note: File paths for good/bad samples and target folder need to be configured in the `__main__` section before execution.
 

2. **Annotation**

`drawPerspective.py` Summary
Purpose: Processes image segments for perspective transformation
Key Functions:  
1. `process_coordinates()`: Reduces polygon points to 4 corners via line intersection  
2. `line_intersection()`: Calculates intersection of two lines  
3. `draw_heatmap_on_building()`: Warps heatmap images onto building photos using homography  
4. `read_segments_coordinates()`: Imports segment coordinates from CSV  
5. `write_segments_to_csv()`: Exports processed coordinates  

Workflow:  
1. Input: CSV with segment coordinates  
2. Process: Reduce polygons to quadrilaterals  
3. Output: New CSV + heatmap-overlaid building image  

---

`paint.html` Summary
Interactive Tool for Image Annotation 
Features:  
- Coordinate system with (0,0) at top-left corner  
- Create polygonal segments/layers  
- Add text boxes (font size adjustable)  
- Draw bidirectional arrows  
- Zoom/pan functionality  
- Layer management (rename/delete)  

Output:  
1. CSV with segment coordinates  
2. Annotated PNG image  

Key Improvements:  
- Fixed coordinate export  
- Added text/arrow annotations  
- Enhanced UI with gradient backgrounds  

Workflow: Upload → Annotate → Export



3. **Possible Denoise Strategy**



`timedff.py`
A program that performs time-delay estimation and voice activity detection (VAD) on two audio signals recorded by two microphones. The program uses the cross-correlation method to estimate the time delay between the two signals, and then filters out events with a time delay less than a specified threshold.

The main features of `timedff.py` include:

1. **Voice Activity Detection**: The program uses a dynamic threshold based on the signal energy envelope to detect active speech segments in the audio.
2. **Time Delay Estimation**: The program calculates the time delay between the two microphone signals using cross-correlation, and filters out events with a delay less than a specified threshold.
3. **Configurable Parameters**: The program allows you to configure various parameters, such as the microphone distance, time delay threshold, and VAD threshold.

`cut_freq.py`
A program that removes a specific frequency range from an audio signal. The program uses a Butterworth bandpass filter to remove the specified frequency range.

The main features of `cut_freq.py` include:

1. **Frequency Removal**: The program can remove a specific frequency range from an audio signal, preserving the rest of the audio.
2. **Configurable Frequency Range**: The program allows you to specify the lower and upper cutoff frequencies for the bandpass filter.
3. **Error Handling**: The program includes error handling to gracefully handle invalid input files or filter parameters.



4. **Audio Samples**
    Some audio samples with different wall condition and fan level



