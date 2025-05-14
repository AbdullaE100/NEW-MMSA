#!/usr/bin/env python3

"""
Run the MMSA interface in local-only mode (no public URL)
This script simplifies running the interface without exposing it to the internet.
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mmsa_local.log')
    ]
)
logger = logging.getLogger('mmsa_local')

def kill_existing_processes():
    """Kill any existing MMSA processes that might be running"""
    try:
        # Check if any related processes are running
        subprocess.run(["pkill", "-f", "mmsa_interface"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if port 7860 is in use
        try:
            result = subprocess.run(["lsof", "-i", ":7860", "-t"], 
                                   capture_output=True, text=True)
            
            if result.stdout.strip():
                logger.info("Port 7860 is in use. Killing process...")
                subprocess.run(["kill", "-9", result.stdout.strip()])
                logger.info("Process killed.")
        except subprocess.CalledProcessError:
            # This is actually good - means port is not in use
            logger.info("Port 7860 is available")
            
    except Exception as e:
        logger.error(f"Error checking/killing processes: {str(e)}")

def main():
    """Run the MMSA interface in local-only mode"""
    # First kill any existing processes
    kill_existing_processes()
    
    # Display startup banner
    print("\n" + "="*60)
    print(" MMSA LOCAL INTERFACE LAUNCHER ")
    print("="*60)
    print("\nStarting the MMSA interface in local-only mode...")
    print("This version does NOT create a public URL and keeps your analysis private.")
    
    # Build the command with explicit no-share option
    cmd = [
        sys.executable,  # Current Python executable
        "mmsa_interface_with_shap.py",
        "--no-share",
        "--port", "7860"
    ]
    
    # Run the MMSA interface with the no-share flag
    try:
        process = subprocess.Popen(cmd)
        
        # Display information to the user
        print("\n" + "="*60)
        print("\nYour MMSA interface is now running locally at:")
        print("\n    http://localhost:7860")
        print("\nThis interface is ONLY accessible on your local machine.")
        print("No public URL is being generated.")
        print("\nPress Ctrl+C to stop the interface when you're done.")
        print("\n" + "="*60)
        
        # Wait for the process to complete or user interrupt
        process.wait()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            # Force kill if needed
            process.kill()
        print("MMSA interface stopped.")
    
    except Exception as e:
        logger.error(f"Error launching MMSA interface: {str(e)}")
        print(f"\nError launching MMSA interface: {str(e)}")
        print("Please check the mmsa_local.log file for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 