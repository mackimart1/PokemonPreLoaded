import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageTk
import tkinter as tk
import time

class ComputerVision:
    def __init__(self, root):
        self.screen_mirror = ScreenMirror(root)
        self.screen_width = 800
        self.screen_height = 600
        self.max_objects = 100

    def capture_screenshot(self):
        return self.screen_mirror.capture_screen()

    def extract_game_state(self, screenshot):
        # Convert PIL Image to numpy array
        screenshot_np = np.array(screenshot)
        # Convert RGB to BGR (OpenCV format)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        # Implement game state extraction logic here
        # This is a placeholder and should be adapted to your specific game
        gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
        return {"average_intensity": np.mean(gray)}

    def process_screenshot(self):
        screenshot = self.capture_screenshot()
        game_state = self.extract_game_state(screenshot)
        return game_state

class ScreenMirror:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Mirror")
        
        # Remove window decorations
        self.root.overrideredirect(True)
        
        # Get screen dimensions
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        
        # Set window dimensions (adjust as needed)
        self.window_width = self.screen_width // 4
        self.window_height = self.screen_height // 4
        
        # Position window at bottom center
        x = (self.screen_width - self.window_width) // 2
        y = self.screen_height - self.window_height
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for dragging the window
        self.root.bind("<ButtonPress-1>", self.start_move)
        self.root.bind("<ButtonRelease-1>", self.stop_move)
        self.root.bind("<B1-Motion>", self.do_move)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.running = True
        self.root.after(100, self.update_screen)  # Schedule the first update
    
    def capture_screen(self):
        return pyautogui.screenshot()
    
    def update_screen(self):
        if self.running:
            screen = self.capture_screen()
            screen = screen.resize((self.window_width, self.window_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(screen)

            self.canvas.config(width=self.window_width, height=self.window_height)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo

            self.root.after(100, self.update_screen)  # Schedule the next update
    
    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")
    
    def on_closing(self):
        self.running = False
        self.root.destroy()

def start_bot():
    root = tk.Tk()
    cv = ComputerVision(root)
    root.mainloop()

def main():
    start_bot()

if __name__ == "__main__":
    main()