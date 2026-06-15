import streamlit as st
import os
from pathlib import Path

# Import helper functions from your backend file
import image_analyzer

# Configure clear and readable layout
st.set_page_config(page_title="Multimodal Image Analyzer Dashboard", page_icon="🖼️", layout="centered")

st.title("🎯 Assignment: Image Analysis via LLaVA:7b")
st.write("An interactive interface built to fulfill strict evaluation guidelines and structured presentation criteria.")

st.markdown("---")

# Criteria 1: UI Input text box for workspace directories
dir_path_input = st.text_input("📁 Enter Project Directory Path:", value=os.getcwd())
dir_path = Path(dir_path_input)

if dir_path.is_dir():
    # Gather image files matching extensions defined in the runner
    all_files = os.listdir(dir_path)
    valid_images = [
        f for f in all_files 
        if Path(f).suffix.lower() in image_analyzer.IMAGE_EXTENSIONS
    ]
    
    if valid_images:
        # Criteria 2: Clean Dropdown Selection 
        selected_image_name = st.selectbox("🖼️ Select Image for Analysis:", valid_images)
        full_image_path = dir_path / selected_image_name
        
        # Display image directly on screen for optimal UI/UX visual grading
        st.image(str(full_image_path), caption=f"Selected Input: {selected_image_name}", use_container_width=True)
        
        # Track matching txt name dynamically using the runner logic
        matching_txt_file = image_analyzer.get_matching_questions_file(selected_image_name)
        full_txt_path = dir_path / matching_txt_file
        
        # Verify the file system status
        if full_txt_path.exists():
            st.success(f"Validated: Associated question file found (`{matching_txt_file}`)")
            
            # Interactive execution controls layout
            col1, col2 = st.columns(2)
            
            # Use Session State arrays to strictly organize live analytical display blocks
            if "dashboard_results" not in st.session_state:
                st.session_state.dashboard_results = []
            
            # FIX 1: Define output paths globally so they don't disappear on rerun
            output_file_name = "all_answers.txt"
            target_output_path = dir_path / output_file_name
            
            with col1:
                # Criteria 3 & 4: Process questions and format the output correctly
                if st.button("🚀 Run", use_container_width=True):
                    st.session_state.dashboard_results = [] # Clear memory slots for fresh iteration
                    
                    # Provide user feedback while waiting for inference execution loops
                    with st.spinner("Processing questions through LLaVA engine... Please hold."):
                        try:
                            # Fire inference pipeline
                            run_data = image_analyzer.run_image_questions(
                                image_path=str(full_image_path),
                                questions_path=str(full_txt_path),
                                output_path=str(target_output_path)
                            )
                            
                            # Save directly to memory states for live display formatting
                            st.session_state.dashboard_results = run_data
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Execution Error encountered: {e}")
            
            with col2:
                # Criteria 5: Gracefully halt operations on button click
                if st.button("❌ Close", use_container_width=True):
                    st.session_state.dashboard_results = []
                    st.warning("Execution loop stopped cleanly by supervisor command.")
                    st.stop()
            
            st.markdown("---")
            
            # 📊 Information Organization Section (ארגון המידע)
            # Renders text question strings and structural model responses inside neat component boxes
            if st.session_state.dashboard_results:
                st.subheader("📊 Dynamic Analysis Display Dashboard")
                st.info(f"Rendering structured outputs found in: `{output_file_name}`")
                
                for idx, record in enumerate(st.session_state.dashboard_results, 1):
                    # Organizes each query/response set clearly
                    with st.expander(f"🔹 Question {idx}: {record['question']}", expanded=True):
                        st.markdown("**Model Output:**")
                        st.code(record['answer'], language="text")
        else:
            st.error(f"Missing File Error: Expected text prompt structure `{matching_txt_file}` does not exist.")
    else:
        st.info("No supported graphic documents discovered inside this workspace directory.")
else:
    st.error("Invalid Path Error: Set context parameter value pointing toward a real local directory.")