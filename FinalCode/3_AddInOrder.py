import serial
import time
import tkinter as tk
from tkinter import scrolledtext

# Set up the serial connection (adjust port name and baud rate as needed)
arduino = serial.Serial('COM5', 9600)  # Change 'COM5' to your Arduino port
time.sleep(2)  # Wait for the serial connection to establish

# Sensor states: [onion, tomato, spinach, egg] (initially all OFF)
sensor_states = [False, False, False, False]
expected_release_order = [2, 1, 3, 4]  # New correct release order: Sensor 2 -> Sensor 1 -> Sensor 3 -> Sensor 4
current_release_index = 0  # Index to track which sensor should be released next

# GUI Application
class SensorGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Sensor Monitor")
        self.geometry("500x400")

        # Instructions at the top
        self.instructions_label = tk.Label(
            self,
            text="Let's add the ingredients to the pan!\nFirst, add the tomato, then the onion, then the spinach, then the egg.",
            font=("Arial", 12),
            wraplength=480,
            justify="center"
        )
        self.instructions_label.pack(pady=10)

        # Scrolled text widget to display messages
        self.log = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=15)
        self.log.pack(pady=10)

        # Label for status updates
        self.status_label = tk.Label(self, text="Waiting for sensor data...", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # Start the sensor monitoring loop
        self.after(100, self.read_sensor_status)

    def log_message(self, message):
        """Log a message to the GUI."""
        self.log.insert(tk.END, f"{message}\n")
        self.log.see(tk.END)  # Scroll to the latest message

    def update_status(self, status):
        """Update the status label."""
        self.status_label.config(text=status)

    def read_sensor_status(self):
        """Read data from the Arduino and update the GUI."""
        global sensor_states, expected_release_order, current_release_index

        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()  # Read and decode data
            if data.startswith("TOUCH"):
                sensor_num = int(data.split()[1]) - 1  # Get the sensor number (0-indexed)
                if not sensor_states[sensor_num]:  # If the sensor wasn't already ON
                    sensor_states[sensor_num] = True  # Mark the sensor as ON
            elif data.startswith("RELEASE"):
                sensor_num = int(data.split()[1]) - 1  # Get the sensor number (0-indexed)
                if sensor_states[sensor_num]:  # If the sensor was ON
                    if sensor_num + 1 == expected_release_order[current_release_index]:
                        sensor_states[sensor_num] = False  # Mark the sensor as OFF
                        current_release_index += 1
                        
                        # Custom messages for each sensor release
                        if sensor_num + 1 == 2:
                            self.log_message("Correct Order: Tomatoes added to pan")
                        elif sensor_num + 1 == 1:
                            self.log_message("Correct Order: Onions added to pan")
                        elif sensor_num + 1 == 3:
                            self.log_message("Correct Order: Spinach added to pan")
                        elif sensor_num + 1 == 4:
                            self.log_message("Correct Order: Egg added to pan")
                        
                        # Send success signal to Arduino
                        arduino.write(b"SUCCESS\n")
                    else:
                        self.log_message("Wrong ingredient!")
                        arduino.write(b"WRONG\n")
                        self.update_status("Try adding ingredients again in the correct order!")

                # If all sensors are turned off, check the order
                if all(not state for state in sensor_states):
                    if current_release_index == 4:  # All sensors turned off in the correct order
                        self.log_message("All ingredients added in correct order!")
                        self.update_status("All ingredients added in correct order!")
                        arduino.write(b"SUCCESS\n")
                    else:
                        self.update_status("Not all sensors turned off in the correct order!")

            elif data == "All sensors activated!":
                self.log_message("All sensors are ON!")
                sensor_states = [True, True, True, True]
                current_release_index = 0
            elif data == "All sensors deactivated!":
                self.log_message("All sensors are OFF!")
                sensor_states = [False, False, False, False]
                current_release_index = 0
            else:
                self.log_message(f"Unknown data received: {data}")

        # Schedule the next read
        self.after(100, self.read_sensor_status)


# Start the GUI
if __name__ == "__main__":
    app = SensorGUI()
    app.mainloop()
