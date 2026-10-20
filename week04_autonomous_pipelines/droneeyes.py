import cv2
from djitellopy import Tello

def main():
    t = Tello()
    t.connect()
    
    battery = t.get_battery()
    print(f'Battery: {battery}%')
    
    print("Starting video stream. Press 'q' to quit (ensure the video window is focused).")
    t.streamon()

    try:
        while True:
            # Get the frame from the drone's camera
            img = t.get_frame_read().frame
            
            # Display the resulting frame
            cv2.imshow("Tello Live View", img)
            
            # Wait for 1 ms and check if 'q' is pressed to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupt received, stopping...")
    finally:
        # Clean up resources
        t.streamoff()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
