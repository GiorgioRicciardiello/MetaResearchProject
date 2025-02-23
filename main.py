import subprocess
import sys
import os

def run_scripts_via_subprocess():
    # Construct paths to your scripts
    step1_script = os.path.join("notebooks", "MetaResearchStep1_FindPapers.py")
    step2_script = os.path.join("notebooks", "MetaResearchStep2_GetFullAbstracts.py")

    # Run Step1
    print("Running Step1: FindPapers...")
    subprocess.run([sys.executable, step1_script], check=True)

    # Run Step2
    print("Running Step2: GetFullAbstracts...")
    subprocess.run([sys.executable, step2_script], check=True)

if __name__ == "__main__":
    run_scripts_via_subprocess()
