import PyInstaller.__main__
import os
import shutil

def build():
    # Check if model exists
    if not os.path.exists('model.p'):
        print("Error: model.p not found! Please train your model first.")
        return

    print("Starting build process...")
    
    # PyInstaller arguments
    params = [
        'app.py',                     # Entry point
        '--name=SensiSign_AI',        # Name of the exe
        '--onefile',                  # Single file executable
        '--noconsole',                # Don't show terminal
        '--icon=static/app_icon.ico', # App icon
        '--add-data=templates;templates', # Include templates
        '--add-data=static;static',       # Include static assets
        '--add-data=model.p;.',           # Include the trained model
        '--collect-all=mediapipe',     # Crucial for Mediapipe
        '--clean',                    # Clear cache
    ]

    PyInstaller.__main__.run(params)
    print("\n" + "="*50)
    print("BUILD COMPLETE!")
    print("Your app is located in the 'dist' folder.")
    print("="*50)

if __name__ == "__main__":
    build()
