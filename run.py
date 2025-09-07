import os
import sys
import subprocess

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(current_dir, 'backend')
    
    # Check if Python is installed
    try:
        subprocess.run([sys.executable, '--version'], check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Error: Python is not installed or not in PATH.")
        return
    
    # Check if requirements are installed
    print("Checking and installing requirements...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', os.path.join(backend_dir, 'requirements.txt')],
            check=True
        )
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements.")
        return
    
    # Run the Flask application
    print("Starting the Deepfake Detector application...")
    try:
        os.chdir(backend_dir)
        env = os.environ.copy()
        env['FLASK_APP'] = 'app.py'
        env['FLASK_ENV'] = 'development'
        subprocess.run(
            [sys.executable, '-m', 'flask', 'run', '--host=0.0.0.0', '--port=5000'],
            env=env
        )
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"Error: Failed to start the application: {str(e)}")

if __name__ == "__main__":
    main()