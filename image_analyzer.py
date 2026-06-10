import os
import ollama

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

def process_image_questions(image_path, questions_path, output_file):
    """
    Reads the input text file questions, queries LLaVA, and saves 
    the response in the exact format required.
    """
    if not os.path.exists(image_path) or not os.path.exists(questions_path):
        print(f"Error: Missing files for {image_path} or {questions_path}")
        return

    # Read questions from text file line by line
    with open(questions_path, 'r', encoding='utf-8') as q_file:
        questions = [line.strip() for line in q_file if line.strip()]

    # Append results to the final output file
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for question in questions:
            prompt = get_optimized_prompt(os.path.basename(image_path), question)
            
            try:
                # Query the local LLaVA model via Ollama
                response = ollama.generate(
                    model='llava:7b',
                    prompt=prompt,
                    images=[image_path]
                )
                
                # Enforce the strict 400-character cap from the guidelines
                raw_answer = response.get('response', '').strip()
                truncated_answer = raw_answer[:400]
                
                # Write matching the presentation format guidelines
                out_file.write(f'picture: "{os.path.basename(image_path)}"\n')
                out_file.write(f'question: "{question}"\n')
                out_file.write(f'answer: "{truncated_answer}"\n\n')
                
            except Exception as e:
                print(f"Error processing question '{question}': {e}")


def run_pipeline():
    # UPDATED: Using your exact file names from your workspace
    tasks = [
        {
            "img": "drawing_Robot Chef in Cozy Kitchen.png", 
            "txt": "drawing_Robot Chef in Cozy Kitchen.txt"
        },
        {
            "img": "flowchart_Bank Loan.png", 
            "txt": "flowchart_Bank Loan.txt"
        },
        {
            "img": "text_Helens Summer Afternoon Story.png", 
            "txt": "text_Helens Summer Afternoon Story.txt"
        }
    ]
    
    output_filename = "all_answers.txt"
    
    # Reset output file if it already exists from a previous run
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    print("Starting image analysis batch...")
    for task in tasks:
        print(f"Processing {task['img']}...")
        process_image_questions(task['img'], task['txt'], output_filename)
    print(f"Done! Results successfully saved to {output_filename}")


if __name__ == "__main__":
    run_pipeline()