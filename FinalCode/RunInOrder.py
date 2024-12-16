import subprocess
import time
import shlex

def upload_arduino_code(sketch_path, port, fqbn):
    """
    Compile and upload Arduino code using arduino-cli.
    """
    try:
        # Compile the Arduino sketch
        print(f"Compiling Arduino code from {sketch_path}...")
        compile_result = subprocess.run(
            ["arduino-cli", "compile", "--fqbn", fqbn, sketch_path],
            capture_output=True,
            text=True
        )

        if compile_result.returncode != 0:
            print(f"Failed to compile Arduino code:\n{compile_result.stderr}")
            return False
        print("Arduino code compiled successfully!")

        # Upload the compiled Arduino code
        print(f"Uploading Arduino code to {port}...")
        upload_result = subprocess.run(
            ["arduino-cli", "upload", "--fqbn", fqbn, "--port", port, sketch_path],
            capture_output=True,
            text=True
        )

        if upload_result.returncode == 0:
            print("Arduino code uploaded successfully!")
        else:
            print(f"Failed to upload Arduino code:\n{upload_result.stderr}")
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
        # Safely handle paths with spaces using shlex.quote
        safe_script_path = shlex.quote(script_path)
        print(f"Running Python script: {script_path}...")
        result = subprocess.run(
            ["python", safe_script_path],
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
    {"script": "C:/Users/linds/Myproject/A3_413/FinalCode/1_Veg_IDChop.py", "arduino_sketch": "C:/Users/linds/Myproject/A3_413/FinalCode/1_VegIDChop/1_VegIDChop.ino", "port": "COM5", "fqbn": "arduino:avr:mega"},
    {"script": "C:/Users/linds/Myproject/A3_413/FinalCode/2_WhiskSpeed.py", "arduino_sketch": "C:/Users/linds/Myproject/A3_413/FinalCode/2_WhiskSpeed/2_WhiskSpeed.ino", "port": "COM5", "fqbn": "arduino:avr:mega"},
    {"script": "C:/Users/linds/Myproject/A3_413/FinalCode/3_AddInOrder.py", "arduino_sketch": "C:/Users/linds/Myproject/A3_413/FinalCode/3_AddInOrder/3_AddInOrder.ino", "port": "COM5", "fqbn": "arduino:avr:mega"}
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
