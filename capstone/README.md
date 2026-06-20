# 🌿 BioShield AI: Plant Disease Diagnostician

BioShield AI is a premium plant disease detection web application that combines a **YOLOv8-style CSPDarknet backbone** for local texture feature extraction with a **Transformer Self-Attention Encoder** for global structural leaf context. 

Developed in **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 🚀 Features
1. **🔍 Dual-Input Diagnosis:** Drag and drop plant leaf images or take a live specimen photo directly using your webcam.
2. **📋 Actionable Agricultural Remedies:** Provides immediate treatment advice, prevention tips, and chemical/organic control options for diagnosed disease classes.
3. **🧠 Model Architecture & Attention Heatmaps:** Explains how the hybrid network blends convolutional feature maps with self-attention, including a visualization of the attention hotspot locations.
4. **💡 Interactive Demo Mode:** Sidebar selectors let you immediately run case studies (Potato Late Blight, Healthy Pepper, and Non-leaf background rejection) to test model behavior without manual uploads.

---

## 🛠️ Project Structure
* `plant_detection.ipynb` - Jupyter notebook containing the stable pipeline, custom model class, hyperparameter tuning, and final CPU/GPU training code.
* `app.py` - Main Streamlit web application.
* `model_definition.py` - Contains the custom Keras layers (`Cbs_Block`, `C2f_Block`, `Transformer_Encoder`, etc.) and the optimal model builder for clean importing.
* `requirements.txt` - Python package dependencies.
* `models/` - Location to store the final weights and hyperparameter config file:
  * `models/final_plant_disease_detection_model.weights.h5`
  * `models/best_hps.json`

---

## 💻 Setup & Run Instructions

### 1. Install Dependencies
Make sure your virtual environment is active and run:
```bash
pip install -r requirements.txt
```

### 2. Export Model Data (Done after Training finishes)
Once your training script finishes, ensure that your training output files are saved under the `models/` directory:
* `models/best_hps.json`
* `models/final_plant_disease_detection_model.weights.h5`

*Note: If these files are not present in your local folder yet (e.g. while your model is still training overnight), the app will automatically run in **Demo Simulation Mode** so you can interact with the interface.*

### 3. Launch the Streamlit App
Run the following command in your terminal to start the local web server:
```bash
streamlit run app.py
```
This will automatically open the web app in your default browser at `http://localhost:8501`.
