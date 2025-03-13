import vertexai
import asyncio
import pandas as pd
import json
import re

from vertexai.generative_models import GenerativeModel

PROJECT_ID = "dotted-task-453409-s2"
vertexai.init(project=PROJECT_ID, location="us-central1")

# Define the model
GEMI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
GEMI_2_0_FLASH = "gemini-2.0-flash-001"

current_model = GEMI_2_0_FLASH_LITE;

model = GenerativeModel(current_model)

input_file_name = 'dataset_head_25.csv'
output_file_name = 'dataset_head_25_2.0_lite_processed.csv'


def generate_prompt(abstract):
    return f"""
    Evaluate this abstract of the paper and give this a score from 1 to 100 if it's related to Civic technology. 
    Civic technology is defined as physical or digital tools that connect citizens with governments or improve 
    public services or enhance community engagement. However, lower the score significantly if it doesn't 
    discuss a case study involving a team working on a software or hardware project.
    
    Output in the JSON format with two fields score and reason. Keep the reason very short.

    Abstract:
    {abstract}
    """

async def process_row(row, index, total, delay=10):
    """
    Calls Vertex AI with simple retry logic to handle ResourceExhausted errors.
    """
    prompt = generate_prompt(row["Abstract"])

    await asyncio.sleep(delay)
    print(f"Processing row {index+1}/{total}")
    response = model.generate_content(prompt)
    
    return response.text  # On success

async def main_process(df):
    scores = []
    reasons = []
    total = len(df)

    # Process rows *sequentially* to keep concurrency low
    for i, row in df.iterrows():
        result = await process_row(row, i, total)
        
        # Result has the format: {score: 50, reason: ""}
        # Get the score from the result and the reason, and append to the results list (2 columns)
        
        print(result)
        
        clean_result = re.sub(r"^```json\s*|```$", "", result.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(clean_result)
            print(data)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
        
        # Extract score and reason
        score = data.get("score")
        reason = data.get("reason")
        
        scores.append(score)
        reasons.append(reason)


    df["Score"] = scores
    df["Reason"] = reasons
    return df


df = pd.read_csv(input_file_name)
df_processed = asyncio.run(main_process(df))
df_processed.to_csv(output_file_name, index=False)