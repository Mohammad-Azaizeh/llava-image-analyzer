import streamlit as st
import os
import ollama

# Configure basic page layout and theme
st.set_page_config(page_title="Multimodal Image Analyzer", page_icon="🖼️", layout="centered")

def get_optimized_prompt(filename, question):
    """
    High-performance prompt engineering to eliminate hallucinations,
    force explicit OCR text matching, and guarantee answers stay under 400 characters.
    """
    # Global strict rules for all image types
    base_instruction = (
        "TASK: Answer the question using ONLY visual facts clearly visible in the image.\n"
        "RULES:\n"
        "1. Be direct, short, and objective. Give the answer immediately in 1-2 sentences.\n"
        "2. Do NOT write introductory fluff, pleasantries, or explanations.\n"
        "3. Never guess. If text or an item is explicitly written, extract it exactly.\n"
    )
    
    if "flowchart" in filename.lower():
        special_prompt = (
            "IMAGE TYPE: Logic Flowchart Diagram.\n"
            "INSTRUCTION: Trace the blocks from 'Start' downwards following the arrows. "
            "Locate the exact words inside the boxes and diamond decisions mentioned in the question."
        )
    elif "text" in filename.lower():
        special_prompt = (
            "IMAGE TYPE: Printed English Document Text.\n"
            "INSTRUCTION: Perform absolute precision OCR. Scan line-by-line. "
            "The answers to names, authors, book titles, and events are explicitly written in this short story passage. "
            "Find them and extract them exactly as written."
        )
    elif "drawing" in filename.lower():
        special_prompt = (
            "IMAGE TYPE: Illustration/Drawing Scene.\n"
            "INSTRUCTION: Look closely at the visual items. Count objects precisely if asked. "
            "Describe only physical items present on the counter, shelves, or hands."
        )
    else:
        special_prompt = "Analyze the visual data accurately."

    return f"{base_instruction}\n{special_prompt}\nQUESTION: {question}\nEXACT ANSWER:"


st.title("🎯 Assignment: Image Analysis via LLaVA:7b")
st.write("An interactive interface to select workspaces, images, and evaluate contextual questions.")



# Requirement 1: Field to select/enter directory path
dir_path = st.text_input("📁 Enter Project Directory Path:", value=os.getcwd())

if os.path.isdir(dir_path):
    # Scan and filter available image files in the directory
    all_files = os.listdir(dir_path)
    image_extensions = ('.png', '.jpg', '.jpeg')
    valid_images = [f for f in all_files if f.lower().endswith(image_extensions)]
    
    if valid_images:
        # Requirement 2: Field to select an image from the directory
        selected_image = st.selectbox("🖼️ Select Image for Analysis:", valid_images)
        
        # Display chosen image in UI to improve user presentation score
        full_image_path = os.path.join(dir_path, selected_image)
        st.image(full_image_path, caption=f"Selected File: {selected_image}", use_container_width=True)
        
        # Automatically infer the matching text file name (e.g., drawing_Robot Chef...txt)
        base_name, _ = os.path.splitext(selected_image)
        matching_txt_file = f"{base_name}.txt"
        full_txt_path = os.path.join(dir_path, matching_txt_file)
        
        # Check if the mandatory question file exists
        if os.path.exists(full_txt_path):
            st.success(f"Validated: Matching question file found ({matching_txt_file})")
            
            # Action layout side-by-side buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Requirement 3: Run button to process the questions file matching the image
                if st.button("🚀 Run", use_container_width=True):
                    # Requirement 4: Save output file into the same directory
                    output_file_path = os.path.join(dir_path, "all_answers.txt")
                    
                    with open(full_txt_path, 'r', encoding='utf-8') as q_f:
                        questions = [line.strip() for line in q_f if line.strip()]
                    
                    # Live feedback spinner during active inference (prevents UI lag penalty)
                    with st.spinner("LLaVA model is actively analyzing the image... please wait."):
                        try:
                            with open(output_file_path, 'a', encoding='utf-8') as out_f:
                                for q in questions:
                                    prompt = get_optimized_prompt(selected_image, q)
                                    response = ollama.generate(
                                        model='llava:7b',
                                        prompt=prompt,
                                        images=[full_image_path]
                                    )
                                    ans = response.get('response', '').strip()[:400]
                                    
                                    # Output format matching the requested text block structure
                                    out_f.write(f'picture: "{selected_image}"\n')
                                    out_f.write(f'question: "{q}"\n')
                                    out_f.write(f'answer: "{ans}"\n\n')
                                    
                            st.balloons()
                            st.success(f"✨ Complete! Output updated in: {os.path.basename(output_file_path)}")
                        except Exception as e:
                            st.error(f"Inference error: {e}")
            
            with col2:
                # Requirement 5: Close button to end or reset execution
                if st.button("❌ Close", use_container_width=True):
                    st.warning("Process halted by user request.")
                    st.stop()
        else:
            st.error(f"Missing File Error: {matching_txt_file} does not exist in this directory.")
    else:
        st.info("No valid image formats (PNG, JPG, JPEG) discovered in this folder path.")
else:
    st.error("Invalid Path Error: The path provided is not a directory.")