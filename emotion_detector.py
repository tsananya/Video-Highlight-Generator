from deepface import DeepFace
import cv2
import pandas as pd
import time
import os

def get_emotional_score(video_path, max_frames=50):
    """
    Analyzes a video for dominant facial emotions and assigns a score.
    
    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to process for speed.
        
    Returns:
        dict: Summary of detected emotions and a total score.
    """
    
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video file."}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Determine frame sampling (only analyze a manageable number of frames)
    # This prevents the process from taking hours for long videos.
    if frame_count > max_frames:
        frame_interval = frame_count // max_frames
    else:
        frame_interval = 1
        
    # 3. Define high-value emotions
    high_value_emotions = ['happy', 'surprise', 'fear']
    emotion_tally = {e: 0 for e in high_value_emotions}
    total_analyzed_frames = 0
    
    # 4. Process frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames based on the interval
        if i % frame_interval != 0:
            continue
            
        total_analyzed_frames += 1

        try:
            # DeepFace analysis (action='emotion')
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False # Detects faces even if partially visible
            )
            
            # Check if a face was detected and emotions analyzed
            if result and isinstance(result, list) and 'dominant_emotion' in result[0]:
                dominant_emotion = result[0]['dominant_emotion']
                if dominant_emotion in high_value_emotions:
                    emotion_tally[dominant_emotion] += 1
                    
        except Exception:
            # Silently skip frames where no face is detected or detection fails
            pass

    cap.release()

    # 5. Calculate Final Score
    # The score is the sum of high-value emotion tallies
    emotional_score = sum(emotion_tally.values())
    
    # 6. Prepare the final summary
    emotion_summary = {
        "analyzed_frames": total_analyzed_frames,
        "happy": emotion_tally['happy'],
        "surprise": emotion_tally['surprise'],
        "excitement_score": emotional_score,
        "detection_success_rate": f"{total_analyzed_frames / (frame_count / frame_interval) * 100:.1f}%"
    }

    return emotion_summary

if __name__ == '__main__':
    # This block is for testing and requires a video path
    # You will need to uncomment this and provide a path to test independently:
    # dummy_video_path = "path/to/your/video.mp4"
    # if os.path.exists(dummy_video_path):
    #     print(get_emotional_score(dummy_video_path))
    # else:
    #     print("Test video not found.")
    pass