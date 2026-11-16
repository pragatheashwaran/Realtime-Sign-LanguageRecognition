from PIL import Image
import os

def convert_to_ico():
    img_path = 'static/icon.png'
    ico_path = 'static/app_icon.ico'
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
        print(f"Converted {img_path} to {ico_path}")
    else:
        print("Icon not found!")

if __name__ == "__main__":
    convert_to_ico()
