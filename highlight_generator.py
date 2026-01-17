from moviepy.editor import VideoFileClip, concatenate_videoclips
import pandas as pd
import os

# Helper function to convert HH:MM:SS.mmm string to total seconds (float)
def time_string_to_seconds(time_str):
    """Converts a time string (HH:MM:SS.mmm) to total seconds (float)."""
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except Exception:
        # Fallback for unexpected format, though shouldn't happen with scorer.py output
        return 0.0


def generate_highlight_video(video_path, scored_segments_df, top_n=3, output_filename="highlight_reel.mp4"):
    """
    Selects the top N highest-scoring segments and stitches them into a new video.
    """

    # 1. Select the top N segments based on the score
    # Filter for segments with a score > 0 and take the top N
    top_segments = scored_segments_df[scored_segments_df['Highlight_Score'] > 0].head(top_n)

    if top_segments.empty:
        print("No segments with a score > 0 were found to generate a highlight reel.")
        return None

    # 2. Extract clip objects for each top segment
    clip_list = []

    try:
        # Load the original clip ONCE
        original_clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error loading original video: {e}")
        return None

    for index, segment in top_segments.iterrows():
        # CONVERSION: Convert time strings (HH:MM:SS.mmm) to seconds (float) for MoviePy
        start_time_sec = time_string_to_seconds(segment['Start_Time'])
        end_time_sec = time_string_to_seconds(segment['End_Time'])

        # Clip the original video using the timecodes (now guaranteed to be a float)
        try:
            subclip = original_clip.subclip(start_time_sec, end_time_sec)
            clip_list.append(subclip)
        except Exception as e:
            print(f"Error creating subclip for segment {index}: {e}")
            pass # Skip this segment if it fails


    if not clip_list:
        original_clip.close()
        return None

    # 3. Concatenate (stitch) the clips together
    print(f"Concatenating {len(clip_list)} segments into the final reel...")
    final_clip = concatenate_videoclips(clip_list)

    # 4. Write the final video file
    output_path = os.path.join(os.path.dirname(video_path), output_filename)
    
    # Use a safe output path, adjusting for temp directory naming
    output_directory = os.path.dirname(video_path)
    final_output_path = os.path.join(output_directory, output_filename)


    # Write the final video file (using logger=None to prevent verbose output)
    final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac', logger=None)
    
    # 5. Clean up
    original_clip.close() # Close the original clip object
    final_clip.close() # Close the final clip object

    return final_output_path

if __name__ == '__main__':
    # ... (rest of the __main__ block is unchanged) ...
    pass