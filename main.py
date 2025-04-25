#Install DeepFace
!pip install deepface

#Import required libraries
from google.colab import files
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

#Stylish header and custom branding
display(HTML("""
<style>
    body {
        background-color: #0d0d0d;
        color: #00ffcc;
        font-family: 'Courier New', monospace;
    }
    .widget-label { color: #00ffcc !important; }
    .output_subarea {
        background: #1a1a1a !important;
        color: #00ffcc !important;
    }
    button {
        background-color: #00ffcc !important;
        color: #0d0d0d !important;
        font-weight: bold;
    }
</style>
<h2 style="color:#00ffcc; text-align:center;">SCHOOL EMOTIONS CHECKING SYSTEM</h2>
<p style="text-align:center;">Upload an image and detect emotions (Happy / Sad / Angry / Neutral / Surprise)</p>
<p style="text-align:center; color:#cccccc; font-size:14px;">Made by Mahin Hasnat</p>
"""))

#Uploading image files
print("Upload up to 5 images:")
uploaded = files.upload()

#Dropdown for selecting uploaded images
image_files = list(uploaded.keys())
dropdown = widgets.Dropdown(
    options=image_files,
    description='Image:',
    layout=widgets.Layout(width='50%')
)

#Buttons for scanning and going back
scan_button = widgets.Button(
    description='Scan Emotion',
    layout=widgets.Layout(width='50%'),
    button_style='success'
)

back_button = widgets.Button(
    description='Select Another Image',
    layout=widgets.Layout(width='50%'),
    button_style='info'
)

#Output widget and allowed emotions
output = widgets.Output()
allowed_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']

#Emotion analysis function
def analyze_emotion(b):
    with output:
        clear_output()
        image_path = dropdown.value
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ Failed to read image: {image_path}")
            return

        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']
            filtered = {e: emotions[e] for e in allowed_emotions}
            dominant = max(filtered, key=filtered.get)

            # Display result
            plt.figure(figsize=(4, 4))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"DETECTED: {dominant.upper()}", color="#00ffcc", fontsize=14)
            plt.axis('off')
            plt.show()

            # Show back button after result
            display(back_button)

        except Exception as e:
            print(f"⚠️ Error: {e}")

#Go back to dropdown Menue
def back_to_selection(b):
    with output:
        clear_output()
        display(widgets.VBox([
            widgets.HTML("<h4 style='color:#00ffcc;'>Select Image:</h4>"),
            dropdown,
            widgets.HTML("<br>"),
            scan_button
        ]))

#Button event binding
scan_button.on_click(analyze_emotion)
back_button.on_click(back_to_selection)

#Initial display layout Making 
display(widgets.VBox([
    widgets.HTML("<h4 style='color:#00ffcc;'>Select Image:</h4>"),
    dropdown,
    widgets.HTML("<br>"),
    scan_button,
    output
]))