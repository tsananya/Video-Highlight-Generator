from scenedetect import ContentDetector, open_video, SceneManager
import pandas as pd
import tempfile
import os

def get_scene_cuts(video_path):
    """Detects scene cuts using the ContentDetector and returns a DataFrame of timecodes."""
    
    # 1. Prepare for detection
    # Note: We don't strictly need a stats file for this demo, but it's good practice.
    
    # 2. Open the video and set up the manager
    try:
        video = open_video(video_path)
        scene_manager = SceneManager(stats_manager=None)
    except Exception as e:
        return pd.DataFrame([{"Error": f"Failed to open video: {e}"}])

    # 3. Add the ContentDetector (detects sudden changes in content/pixels)
    # Threshold 27 is a good starting point for fast cuts.
    scene_manager.add_detector(ContentDetector(threshold=27))

    # 4. Process the video
    # show_progress=True makes it display a progress bar in the terminal during detection
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # 5. Convert scene list to a structured format (list of dicts)
    data = []
    for i, scene in enumerate(scene_list):
        data.append({
            "Segment": i + 1,
            "Start_Time": str(scene[0].get_timecode()),
            "End_Time": str(scene[1].get_timecode())
        })

    # 6. Return the data as a table (Pandas DataFrame)
    return pd.DataFrame(data).set_index("Segment")

# You can remove or comment out this block later, it's just for testing this file independently.
if __name__ == '__main__':
    # You would need to put a real video path here to test this file alone
    # print(get_scene_cuts("path/to/your/video.mp4"))
    pass