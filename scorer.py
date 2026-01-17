import pandas as pd
from datetime import datetime, timedelta

# Helper function to convert time strings (like "00:00:04.500") to seconds for calculation
def time_to_seconds(time_str):
    # This assumes the format HH:MM:SS.mmm, which SceneDetect and Librosa use
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
    else: # Handle cases where hour might be missing if librosa or scenedetect output varies
        m, s = map(float, parts)
        h = 0.0
    return h * 3600 + m * 60 + s

def calculate_highlight_scores(segments_df, peaks_list, emotional_summary):
    """
    Calculates the importance score for each visual segment based on audio peak density.
    
    Args:
        segments_df (pd.DataFrame): DataFrame of visual segments (from scene_detector.py)
        peaks_list (list): List of dictionaries for audio peaks (from audio_analyzer.py)
    
    Returns:
        pd.DataFrame: Segments DataFrame with a new 'Highlight_Score' column.
    """
    
    # 1. Convert audio peak times (strings) to seconds
    peaks_seconds = [time_to_seconds(peak['time']) for peak in peaks_list if 'time' in peak]
    
    # 2. Prepare the results list
    scored_segments = []
    
    # 2. Get the overall emotional score from the summary
    overall_emotional_score = emotional_summary.get('excitement_score', 0)
    
    # 3. Define the weight for emotional impact. (e.g., 5 points per high-emotion frame)
    EMOTION_WEIGHT = 5 
    
    # 4. Calculate the bonus score based on overall emotion (normalizing by frames analyzed)
    frames_analyzed = emotional_summary.get('analyzed_frames', 1)
    
    # Bonus score is the average high-emotion per frame, multiplied by the segment duration
    # This rewards the video's emotional intensity.
    emotion_bonus_per_second = (overall_emotional_score / frames_analyzed) * EMOTION_WEIGHT 


    # 5. Iterate through each visual segment
    for index, segment in segments_df.iterrows():
        segment_start = time_to_seconds(segment['Start_Time'])
        segment_end = time_to_seconds(segment['End_Time'])
        segment_duration = segment_end - segment_start
        
        # 6. Score Logic 1: Audio Peak Density (Primary Score)
        peak_count = 0
        for peak_time in peaks_seconds:
            if segment_start <= peak_time < segment_end:
                peak_count += 1
        
        # 7. Score Logic 2: Emotional Bonus (Secondary Score)
        # We assume emotional impact applies across the duration of the clip
        emotion_bonus = emotion_bonus_per_second * segment_duration
        
        # 8. Calculate Final Fused Score
        final_score = peak_count + emotion_bonus
        
        # 9. Create a new dictionary for the scored segment
        scored_segments.append({
            'Segment': index,
            'Start_Time': segment['Start_Time'],
            'End_Time': segment['End_Time'],
            'Duration': round(segment_duration, 2),
            'Audio_Peaks': peak_count,
            'Emotion_Bonus': round(emotion_bonus, 2),
            'Highlight_Score': round(final_score, 2)
        })
        
    # 6. Convert to DataFrame, sort by score (highest first), and return
    scored_df = pd.DataFrame(scored_segments).set_index('Segment')
    # Sort by the score to clearly rank the most important segments
    scored_df = scored_df.sort_values(by='Highlight_Score', ascending=False)
    
    return scored_df


# Example usage (for testing purposes, should be called from app.py)
if __name__ == '__main__':
    # Dummy segment data (must match output of scene_detector.py)
    dummy_segments = pd.DataFrame([
        {'Start_Time': '00:00:00.000', 'End_Time': '00:00:10.000'},
        {'Start_Time': '00:00:10.000', 'End_Time': '00:00:20.000'}
    ])
    # Dummy peak data (must match output of audio_analyzer.py)
    dummy_peaks = [
        {'time': '00:00:05.500', 'score': 90},  # Falls in Segment 1
        {'time': '00:00:15.200', 'score': 95},  # Falls in Segment 2
        {'time': '00:00:16.800', 'score': 88}   # Falls in Segment 2 (Score 2)
    ]
    
    scored_results = calculate_highlight_scores(dummy_segments, dummy_peaks)
    print(scored_results)

# Expected output shows Segment 2 (score 2) is higher than Segment 1 (score 1)