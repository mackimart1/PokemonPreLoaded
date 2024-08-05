import pyautogui
import time

class SimulatedKeyboard:
    def __init__(self, custom_keys=None):
        self.keys = custom_keys if custom_keys else [
            # Default key set
            'up', 'down', 'left', 'right',  # Arrow keys
            'z', 'x', 'c', 'v'  # Action buttons
            # Menu buttons
            '1', '2', '3', '4', '5', '6', # Number keys
            # Utility keys
        ]
        self.key_hold_duration = 0.1  # Default key hold duration in seconds

    def press_key(self, action_index):
        if 0 <= action_index < len(self.keys):
            key = self.keys[action_index]
            pyautogui.keyDown(key)
            time.sleep(self.key_hold_duration)
            pyautogui.keyUp(key)
            print(f"Pressed key: {key}")
        else:
            print(f"Invalid action index: {action_index}")

    def hold_key(self, action_index, duration):
        if 0 <= action_index < len(self.keys):
            key = self.keys[action_index]
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            print(f"Held key: {key} for {duration} seconds")
        else:
            print(f"Invalid action index: {action_index}")

    def press_combination(self, *action_indices):
        keys = [self.keys[i] for i in action_indices if 0 <= i < len(self.keys)]
        if keys:
            pyautogui.hotkey(*keys)
            print(f"Pressed key combination: {' + '.join(keys)}")
        else:
            print("Invalid key combination")

    def get_key_count(self):
        return len(self.keys)

    def get_key_list(self):
        return self.keys.copy()

    def set_key_hold_duration(self, duration):
        self.key_hold_duration = max(0.05, duration)  # Ensure minimum duration of 0.05 seconds
        print(f"Key hold duration set to {self.key_hold_duration} seconds")

    def type_string(self, string):
        pyautogui.write(string)
        print(f"Typed string: {string}")

    def move_mouse(self, x, y):
        pyautogui.moveTo(x, y)
        print(f"Moved mouse to ({x}, {y})")

    def click_mouse(self, x=None, y=None):
        if x is not None and y is not None:
            pyautogui.click(x, y)
            print(f"Clicked at ({x}, {y})")
        else:
            pyautogui.click()
            print("Clicked at current mouse position")

if __name__ == "__main__":
    # Test the SimulatedKeyboard
    keyboard = SimulatedKeyboard()
    
    print("Testing key press...")
    keyboard.press_key(0)  # Press 'up' key
    
    print("\nTesting key hold...")
    keyboard.hold_key(1, 0.5)  # Hold 'down' key for 0.5 seconds
    
    print("\nTesting key combination...")
    keyboard.press_combination(0, 2, 4)  # Press 'up', 'left', 'a' simultaneously
    
    print("\nTesting custom key hold duration...")
    keyboard.set_key_hold_duration(0.2)
    keyboard.press_key(3)  # Press 'right' key with new hold duration
    
    print("\nTesting string typing...")
    keyboard.type_string("Hello, World!")
    
    print("\nTesting mouse movement and click...")
    keyboard.move_mouse(100, 100)
    keyboard.click_mouse()
    
    print("\nKeyboard simulation test complete.")