import subprocess
import time

def upload_arduino_code(sketch_path, port, fqbn):
    """
    Upload Arduino code using arduino-cli.
    """
    try:
        print(f"Uploading Arduino code from {sketch_path}...")
        result = subprocess.run(
            ["arduino-cli", "upload", "-p", port, "--fqbn", fqbn, sketch_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Arduino code uploaded successfully!")
        else:
            print(f"Failed to upload Arduino code:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Error while uploading Arduino code: {e}")
        return False
    return True

def run_python_script(script_path):
    """
    Run a Python script as a subprocess.
    """
    try:
        print(f"Running Python script: {script_path}...")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Python script executed successfully!")
        else:
            print(f"Python script failed:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Error while running Python script: {e}")
        return False
    return True

# Configuration for Arduino and Python scripts
tasks = [
    {"script": "script1.py", "arduino_sketch": "sketch1", "port": "/dev/ttyUSB0", "fqbn": "arduino:avr:uno"},
    {"script": "script2.py", "arduino_sketch": "sketch2", "port": "/dev/ttyUSB0", "fqbn": "arduino:avr:uno"},
    {"script": "script3.py", "arduino_sketch": "sketch3", "port": "/dev/ttyUSB0", "fqbn": "arduino:avr:uno"}
]

# Main execution loop
for task in tasks:
    if not upload_arduino_code(task["arduino_sketch"], task["port"], task["fqbn"]):
        print("Aborting sequence due to Arduino upload failure.")
        break

    time.sleep(2)  # Wait a moment for the Arduino to initialize (adjust as needed)

    if not run_python_script(task["script"]):
        print("Aborting sequence due to Python script failure.")
        break
