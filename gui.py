import Tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import pickle

class DigitRecognizerApp:
    def __init__(self, master, network_file="data/network.pkl"):
        self.master = master
        self.master.title("Digit Recognizer")
        
        # Load the trained network
        with open(network_file, "rb") as f:
            self.biases, self.weights = pickle.load(f)
        
        # Create Canvas
        self.canvas_size = 280  # Scaled-up canvas (10x zoom for better drawing)
        self.pixel_size = self.canvas_size // 28
        self.canvas = tk.Canvas(self.master, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Create Buttons
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=5)
        
        self.recognize_button = tk.Button(self.master, text="Recognize", command=self.recognize_digit)
        self.recognize_button.grid(row=1, column=1, pady=5)
        
        # Label to show the prediction
        self.result_label = tk.Label(self.master, text="Draw a digit and press Recognize", font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Create a PIL image for capturing drawing
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        """Draw on the canvas and update the internal image"""
        x, y = event.x, event.y
        x1, y1 = (x - self.pixel_size), (y - self.pixel_size)
        x2, y2 = (x + self.pixel_size), (y + self.pixel_size)
        
        # Draw on Tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        
        # Draw on PIL image (downscaling to 28x28)
        self.draw.ellipse([x / self.pixel_size, y / self.pixel_size, (x + self.pixel_size) / self.pixel_size, (y + self.pixel_size) / self.pixel_size], fill=255)

    def clear_canvas(self):
        """Clears the drawing canvas and resets the image"""
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and press Recognize")

    def recognize_digit(self):
        self.image.save("digit.png")

        """Process the drawn image and use the neural network to recognize the digit"""
        img_array = np.array(self.image, dtype=np.float32) / 255.0
        img_array = img_array.reshape(28*28, 1)  # Flatten into (784, 1) input
        
        # Neural network prediction
        output = self.feedforward(img_array)
        predicted_digit = np.argmax(output)

        # Display the result
        self.result_label.config(text="Recognized Digit: {}".format(predicted_digit))

    def feedforward(self, a):
        """Neural network forward pass"""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-z))

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

