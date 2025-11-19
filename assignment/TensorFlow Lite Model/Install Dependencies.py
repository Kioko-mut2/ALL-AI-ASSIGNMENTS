import sys
import subprocess

def install(packages):
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    try:
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Installation failed:", e)
        print("You can retry in a terminal with:")
        print(" ".join(cmd))

if __name__ == "__main__":
    packages = ["tensorflow", "pillow", "numpy"]
    install(packages)
