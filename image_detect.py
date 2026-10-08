from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2

# Hide Tkinter window
root = Tk()
root.withdraw()

# Select image
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected!")
    exit()

# Load model
model = YOLO("best(1).pt")

# Run detection
results = model.predict(image_path, conf=0.5)

# Get detected image
annotated = results[0].plot()

# Show output
cv2.imshow("NEXUS PPE Detection Result", annotated)

# Save output (optional)
cv2.imwrite("output.jpg", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()