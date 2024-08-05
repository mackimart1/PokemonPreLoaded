from pynput import keyboard
from pynput.keyboard import Key, KeyCode, Controller
import time

class KeyMapper:
    def __init__(self):
        self.mapped_keys = {}
        self.mapping_order = ['Up', 'Down', 'Left', 'Right', 'A', 'B', 'Start', 'Select']
        self.current_mapping_index = 0
        self.current_key = None
        self.listener = None
        self.controller = Controller()

    def on_press(self, key):
        if key == Key.enter:
            if self.current_key:
                action = self.mapping_order[self.current_mapping_index]
                self.mapped_keys[action] = self.current_key
                print(f"Mapped '{action}' to key: {self.current_key}")
                self.current_mapping_index += 1
                self.current_key = None
                if self.current_mapping_index < len(self.mapping_order):
                    print(f"\nPress the key for {self.mapping_order[self.current_mapping_index]}...")
                else:
                    print("\nAll keys mapped. Mapping finished.")
                    return False
        elif key == Key.esc:
            print("\nMapping cancelled.")
            return False
        else:
            self.current_key = key
            print(f"Key pressed: {key}. Press Enter to confirm or press another key.")

    def start_mapping(self):
        print("Key mapping started. Press keys in the following order:")
        print(", ".join(self.mapping_order))
        print("Press 'Enter' to confirm each key, or 'Esc' to cancel mapping.")
        print(f"\nPress the key for {self.mapping_order[0]}...")
        
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def get_mapped_keys(self):
        return self.mapped_keys

class SimulatedKeyboard:
    def __init__(self, mapped_keys):
        self.mapped_keys = mapped_keys
        self.keyboard_controller = Controller()

    def press_key(self, action):
        key = self.mapped_keys.get(action)
        if key:
            print(f"Pressing key {key} for action {action}")
            self.keyboard_controller.press(key)
            self.keyboard_controller.release(key)
        else:
            print(f"No key mapped for action {action}")

def main():
    mapper = KeyMapper()
    mapper.start_mapping()
    mapped_keys = mapper.get_mapped_keys()

    if len(mapped_keys) < len(mapper.mapping_order):
        print("Mapping was incomplete. Exiting.")
        return

    print("\nMapped keys:")
    for action, key in mapped_keys.items():
        print(f"{action}: {key}")

    simulated_keyboard = SimulatedKeyboard(mapped_keys)

    print("\nTesting mapped keys. Press 'Esc' to exit.")
    with keyboard.Listener(on_press=lambda k: k == Key.esc) as listener:
        for action in mapper.mapping_order:
            simulated_keyboard.press_key(action)
            time.sleep(0.5)
        listener.join()

if __name__ == "__main__":
    main()