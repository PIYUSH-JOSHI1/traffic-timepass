import subprocess
import threading
import time
import sys

# Global variables to control the process
process = None
running = False

def run_command(command):
    global running
    running = True
    global process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Continuously read output
        while running:
            output = process.stdout.readline()
            if output:
                print(output.decode().strip())  # Print the output to the terminal
            else:
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.stdout.close()

def start_process(command):
    global running
    if not running:
        print("Starting process...")
        threading.Thread(target=run_command, args=(command,), daemon=True).start()
    else:
        print("Process is already running.")

def stop_process():
    global running
    if running:
        running = False
        if process:
            process.terminate()  # Terminate the subprocess
            print("Stopping process...")
    else:
        print("Process is not running.")

def main_menu(command):
    while True:
        print("\nOptions:")
        print("1. Start Process")
        print("2. Stop Process")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")

        if choice == '1':
            start_process(command)
        elif choice == '2':
            stop_process()
        elif choice == '3':
            stop_process()  # Ensure process is stopped before exiting
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    # Replace 'your_command_here' with the command you want to run
    command_to_run = "python your_script.py"  # Example command
    main_menu(command_to_run)
