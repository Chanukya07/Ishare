import time
import sys
import argparse
from pathlib import Path

# Try importing dependencies, handle if missing (though requirements.txt exists now)
try:
    import speech_recognition as sr
    import pyttsx3
except ImportError:
    print("Please install 'speech_recognition' and 'pyttsx3' to use Smart Cane features.")
    sys.exit(1)

from detect import run, ROOT

def main():
    r = sr.Recognizer()
    print("Smart Cane System Initialized.")
    print("Say 'Hello' to start detection for 10 seconds.")
    print("Say 'Exit' to quit.")

    while True:
        try:
            # Listen for a voice command
            with sr.Microphone() as source:
                print("\nListening...")
                # r.adjust_for_ambient_noise(source) # Optional: Adjust for noise
                audio = r.listen(source, timeout=5, phrase_time_limit=5)

            # Try to recognize the speech
            print("Recognizing...")
            try:
                text = r.recognize_google(audio).lower()
                print(f"You said: {text}")
            except sr.UnknownValueError:
                print("Could not understand audio.")
                continue

            if "hello" in text:
                print("Starting object detection...")
                engine = pyttsx3.init()

                # Run detection
                # We use source='0' for webcam by default
                try:
                    run(
                        weights=ROOT / 'yolov9e.pt',
                        source='0',
                        timeout=10,
                        engine=engine,
                        nosave=True # Do not save images by default
                    )
                except Exception as e:
                    print(f"Error during detection: {e}")
                finally:
                    engine.stop()

            elif "exit" in text:
                print("Exiting...")
                break

        except sr.WaitTimeoutError:
            pass # Continue listening
        except sr.RequestError as e:
            print(f"Network error: {e}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # time.sleep(1) # Prevent tight loop on error

if __name__ == "__main__":
    main()
