import streamlit as st
import pandas as pd
import os
import time
import tempfile
from scorer import calculate_highlight_scores
# ----------------------------------------------
# Import the functions you wrote in Steps 4 and 5
# ----------------------------------------------
from scene_detector import get_scene_cuts
from audio_analyzer import get_audio_peaks
from emotion_detector import get_emotional_score
#from highlight_generator import generate_highlight_video # <<< UNCOMMENTED THIS LINE

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Highlights Generator Demo")
st.title("🎬 Smart Video Highlights Generator: Final System Demo")
st.markdown("### Goal: Demonstrate Multimodal Fusion and Final Video Generation")
st.markdown("---")

# --- 1. Video Input and Preprocessing ---
st.header("1. Input & Preprocessing")
st.info("Upload a video file to begin analysis. The system will save it temporarily for processing.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # --- Step 1A: Save the uploaded file temporarily ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Uploaded Video Preview")
        st.video(temp_file_path)
        st.success(f"File saved to temporary path: `{os.path.basename(temp_file_path)}`")

    with col2:
        st.subheader("Video Metadata")
        st.metric(label="File Name", value=uploaded_file.name)
        st.metric(label="File Size", value=f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        st.markdown("---")

    st.markdown("---")

    # --- Step 1B: Analysis Button to trigger the feature extraction ---
    if st.button("▶️ Run Full Multimodal Analysis & Generate Highlight Reel", type="primary"):
        st.toast('Analysis started...', icon='⏳')
        progress_bar = st.progress(0)

        # --- 2. EXECUTE VISUAL ANALYSIS (Scene Cuts) ---
        st.header("2. Visual Feature Extraction: Scene Change Detection")
        st.markdown("The `scene_detector.py` module uses **PySceneDetect/OpenCV** to find abrupt visual cuts.")
        progress_bar.progress(20)
        time.sleep(0.5)

        # Call the function from your scene_detector.py file
        scene_cuts_df = get_scene_cuts(temp_file_path)

        st.success("✅ **Result: Detected Content Segments (Visual Cuts)**")
        st.dataframe(scene_cuts_df, use_container_width=True)

        st.markdown("---")

        # --- 3. EXECUTE AUDITORY ANALYSIS (Audio Peaks) ---
        st.header("3. Auditory Feature Extraction: Audio Peak Analysis")
        st.markdown("The `audio_analyzer.py` module uses **Librosa/Pydub** to find loud, high-energy moments.")
        progress_bar.progress(40)
        time.sleep(0.5)

        # Call the function from your audio_analyzer.py file
        peak_data, energy_df = get_audio_peaks(temp_file_path)

        col_chart, col_data = st.columns(2)

        with col_chart:
            st.subheader("Audio Energy Over Time")
            st.line_chart(energy_df, x='Time (s)', y='Energy')
            st.caption("Spikes indicate high volume or audio events.")

        with col_data:
            st.subheader("Detected Peak Timecodes")
            # Convert list of dicts (peak_data) to a DataFrame for clean display
            peak_df = pd.DataFrame(peak_data)[['time', 'score']]
            st.success("✅ **Result: Timecodes of High-Energy Audio Peaks**")
            st.dataframe(peak_df, use_container_width=True)

        # --- 4. EXECUTE PHASE 3 ANALYSIS (Emotional Score) ---
        st.header("4. Phase 3: Emotional Feature Extraction")
        st.markdown("The system analyzes faces in the video for high-value emotions (Happy, Surprise) to create an overall **Excitement Score**.")
        progress_bar.progress(60)
        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # CALL THE EMOTION DETECTOR FUNCTION
        # ----------------------------------------------------------------------

        emotion_summary = get_emotional_score(temp_file_path)

        if "error" not in emotion_summary:
            st.success("✅ **Result: Emotional Summary**")

            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric(label="Frames Analyzed", value=emotion_summary['analyzed_frames'])
            with col_e2:
                st.metric(label="High-Emotion Frames", value=emotion_summary['excitement_score'])
            with col_e3:
                st.metric(label="Detection Success Rate", value=emotion_summary['detection_success_rate'])

            st.dataframe(pd.DataFrame([emotion_summary]).drop(columns=['analyzed_frames', 'detection_success_rate', 'excitement_score']), use_container_width=True)

        else:
            st.error(f"Emotion Detection Error: {emotion_summary['error']}")

        st.markdown("---")

        # --- 5. EXECUTE PHASE 2 SCORING (Feature Fusion) ---
        st.header("5. Final Phase: Highlight Scoring System (Multimodal Fusion)") # <<< UNCOMMENTED HEADER
        st.markdown("The system now fuses the Visual Segments, Auditory Peaks, and Emotional Data to calculate a final **Highlight Score** for each clip.")
        progress_bar.progress(80)
        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # CALL THE SCORER FUNCTION AND DISPLAY RESULTS
        # ----------------------------------------------------------------------

        # Ensure we have data from all modules before scoring
        if 'scene_cuts_df' in locals() and 'peak_data' in locals():
            try:
                # 1. Call the scorer function
                scored_results_df = calculate_highlight_scores(scene_cuts_df, peak_data, emotion_summary)

                # 2. Display the final ranked list
                st.subheader("Final Ranked Highlights")
                st.success("🏆 The segments are ranked below by their calculated **Highlight Score**.")
                st.dataframe(scored_results_df, use_container_width=True)

                # 3. Extract the Top Highlight
                top_segment = scored_results_df.iloc[0]

                st.markdown("---")
                st.subheader("🥇 Top Highlight Identified:")
                st.info(f"The most engaging segment starts at **{top_segment['Start_Time']}** and ends at **{top_segment['End_Time']}** (Score: {top_segment['Highlight_Score']}).")

            except Exception as e:
                st.error(f"Error during Highlight Scoring: {e}")
        else:
            st.warning("Cannot run Phase 2 scoring: Visual or Auditory data is missing.")


        st.markdown("---") # <<< INSERTION POINT FOR SECTION 6

        # --- 6. PHASE 4: FINAL HIGHLIGHT GENERATION ---
        # st.header("6. Final Phase: Highlight Video Generation")
        # st.markdown("The system selects the highest-scoring segments and stitches them into the final highlight reel.")
        # progress_bar.progress(95)
        
        # # Ensure we have the scored data from the previous step
        # if 'scored_results_df' in locals():
        #     try:
        #         output_video_path = generate_highlight_video(temp_file_path, scored_results_df, top_n=3)

        #         if output_video_path:
        #             st.success(f"🎥 Highlight Video Generated: {os.path.basename(output_video_path)}")

        #             # Create a download button for the user
        #             with open(output_video_path, "rb") as file:
        #                 st.download_button(
        #                     label="Download Highlight Reel",
        #                     data=file,
        #                     file_name=os.path.basename(output_video_path),
        #                     mime="video/mp4"
        #                 )
        #         else:
        #             st.error("Could not generate highlight video. Check terminal for MoviePy errors.")
        #     except Exception as e:
        #         st.error(f"Critical error in video generation (Phase 4): {e}")


        # # --- Update the final conclusion and cleanup ---
        # progress_bar.progress(100)
        # st.balloons()
        # st.toast('Full Project Execution Complete!', icon='✅')
        # st.markdown("---") 


        # --- Conclusion for Phase 1 ---
        st.header("Phase 1 Completion: Foundation Ready")
        st.info(
            """
            This step successfully extracted the two key modalities required for scoring:
            1. **Time-coded Visual Segments** (Scene Cuts)
            2. **Time-coded Auditory Events** (Loudness Peaks)

            This data is now ready to be processed by the **Highlight Scoring System (Phase 2)**.
            """
        )

    # --- Cleanup after processing (good practice) ---
    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
        # We can't delete it right away if Streamlit is still viewing it, but for clean
        # file management, you would typically delete temp files on exit.
        pass