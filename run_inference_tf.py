import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image manipulation

# --- 1. Configuration ---
# Path to the exported Frozen Graph (.pb) file
# Look inside 'exported-models/...' for a file usually named 'snapshot-...pb' or 'frozen_inference_graph.pb'
PB_MODEL_PATH = 'path/to/your/exported/model/frozen_inference_graph.pb'

# Path to an image you want to analyze
IMAGE_PATH = 'path/to/a/test/image.png'

# --- 2. Load the Frozen Graph ---
print(f"Loading model from: {PB_MODEL_PATH}")
def load_frozen_graph(filepath):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_frozen_graph(PB_MODEL_PATH)

# --- 3. Create Session and Get Tensors ---
# We use a TF 1.x style session (compatible with TF 2.x)
sess = tf.compat.v1.Session(graph=graph)

# DeepLabCut standard tensor names for TensorFlow export:
# Input: 'Placeholder:0'
# Output: 'concat_1:0' (contains heatmaps + location refinement)
try:
    input_tensor = graph.get_tensor_by_name('Placeholder:0')
    output_tensor = graph.get_tensor_by_name('concat_1:0')
except KeyError:
    print("Error: Could not find standard DLC tensor names (Placeholder:0, concat_1:0).")
    exit(1)

print(f"Input tensor: {input_tensor}")
print(f"Output tensor: {output_tensor}")

# --- 4. Pre-process the image ---
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit(1)

# Convert BGR to RGB (DLC models expect RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Add batch dimension (1, height, width, 3)
input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)

# --- 5. Run inference ---
print("Running inference...")
output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})

# --- 6. Get and interpret the output ---
print(f"Raw output shape: {output_data.shape}")
print("\n--- SUCCESS ---")
print("The model ran successfully.")
print("The output tensor contains combined Scoremaps (heatmaps) and Location Refinement fields.")
print("You would now need to extract peaks and apply refinement to get (x,y) coordinates.")