import json
from datasets import load_dataset
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Get API key from environment variable
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. Please set it in .env file or environment.")
    import sys
    sys.exit(1)

client = OpenAI(api_key=API_KEY)


prompt_template = """You are an expert debugging assistant with meta-reasoning abilities.   Your task is to help a novice developer identify the most useful past bug report to assist in locating and fixing the CURRENT bug.   

You will evaluate 10 candidate bug reports based on **five reasoning strategies**:

1.    Structural Similarity: Similar stack traces, error messages, or call graph structure.
2.    Module/Component Similarity: Involves the same files, modules, functions, or subsystems.
3.    Symptom Similarity: Similar observable behaviors (e.g., button unresponsive, UI freeze).
4.    Impact Similarity: Affects the same user flows, APIs, or workflows.

Please follow this process:
- Carefully read the CURRENT bug report.
- For each candidate:
  1. Evaluate it on each of the 5 criteria above (rate 1–10).
  2. Select the most helpful report overall.
  3. Provide a short justification for why you chose it as most helpful for fixing and debugging the current bug.

Finally, provide your answer in the following strict format using XML-style tags:

- Your selected candidate must be wrapped with <FINAL CHOICE>CANDIDATE_A/B/C/D/E.</FINAL CHOICE>
- Your justification must be wrapped with <JUSTIFICATION>...</JUSTIFICATION>

For example:
<FINAL CHOICE>CANDIDATE_D</FINAL CHOICE>
<JUSTIFICATION>Candidate D addresses similar components in the `io.fits` module and has a structural similarity to the CURRENT bug, as both involve HDU handling and modifications of the reading mechanisms. The error related to variable management in reading HDUs could provide insights into ensuring the `replace` function's expected behavior. The relevance of shared fixes and similar testing challenges makes it particularly useful for debugging the CURRENT bug.</JUSTIFICATION>

# CURRENT_BUG_REPORT
{current_bug_report}

---

# CANDIDATE_BUG_REPORT_A
## Bug Report:
{bug_report_a}

## Fix Description (PR):
{pr_a}

## Fixed Files:
{file_paths_a}

---

# CANDIDATE_BUG_REPORT_B
## Bug Report:
{bug_report_b}

## Fix Description (PR):
{pr_b}

## Fixed Files:
{file_paths_b}


---

# CANDIDATE_BUG_REPORT_C
## Bug Report:
{bug_report_c}

## Fix Description (PR):
{pr_c}

## Fixed Files:
{file_paths_c}


---

# CANDIDATE_BUG_REPORT_D
## Bug Report:
{bug_report_d}

## Fix Description (PR):
{pr_d}

## Fixed Files:
{file_paths_d}

---

# CANDIDATE_BUG_REPORT_E
## Bug Report:
{bug_report_e}

## Fix Description (PR):
{pr_e}

## Fixed Files:
{file_paths_e}

"""



import re
def extract_final_choice_and_justification(response_str):
   
    match = re.search(r"<FINAL CHOICE>\s*(CANDIDATE_([A-E]))\s*</FINAL CHOICE>", response_str, re.IGNORECASE)
    
    if not match:
        match = re.search(r"<FINAL CHOICE>\s*CANDIDATE_BUG_REPORT_([A-E])\s*</FINAL CHOICE>", response_str, re.IGNORECASE)
        if match:
            final_choice = f"CANDIDATE_{match.group(1).upper()}"
        else:
            match = re.search(r"<FINAL CHOICE>\s*([A-E])\s*</FINAL CHOICE>", response_str, re.IGNORECASE)
            final_choice = f"CANDIDATE_{match.group(1).upper()}" if match else None
    else:
        final_choice = match.group(1).upper()

    justification_match = re.search(r"<JUSTIFICATION>(.*?)</JUSTIFICATION>", response_str, re.DOTALL | re.IGNORECASE)
    justification = justification_match.group(1).strip() if justification_match else None

    return final_choice, justification




def extract_unique_file_paths(similar_bug_items):
    all_paths = set()
    for item in similar_bug_items:
        for change in item.get("changes", []):
            file_path = change.get("file")
            if file_path:
                all_paths.add(file_path)
    return all_paths


def extract_candidate_info(item):
    issue_title = item.get("issue_title") or ""
    issue_body = item.get("issue_body") or ""
    bug_report = issue_title + "\n" + issue_body

    pr_title = item.get("pr_title") or ""
    pr_body = item.get("pr_body") or ""
    pr = pr_title + "\n" + pr_body

    all_paths = set()
    changes = item.get("changes") or []
    for change in changes:
        file_path = change.get("file")
        if file_path:
            all_paths.add(file_path)
    file_paths = all_paths

    return bug_report.strip(), pr.strip(), file_paths


import tiktoken

MAX_CONTEXT_LENGTH = 128000
MAX_TOKENS_PER_REPORT = MAX_CONTEXT_LENGTH // 10  

def num_tokens(message, model="gpt-4o"):
    """Returns the number of tokens used by a message (single or list of messages)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if isinstance(message, list):
        # Accumulate token count from all message contents
        num_tokens = sum(len(encoding.encode(m["content"])) for m in message)
    else:
        # Single string message
        num_tokens = len(encoding.encode(message))
    return num_tokens

def message_too_long(message, model_name="gpt-4o"):
    return num_tokens(message, model=model_name) >= MAX_TOKENS_PER_REPORT

def truncate_by_token_limit(text, max_tokens, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def pad_candidate_bug(candidate_bug_dict, max_candidates=5):
    padded = {}
    for i in range(1, max_candidates + 1):
        key = f"Candidates_{i}"
        if key in candidate_bug_dict:
            padded[key] = candidate_bug_dict[key]
        else:
            padded[key] = {
                "bug_report": "",
                "pr": "",
                "file_paths": ""
            }
    return padded




if __name__ == "__main__":
    # =====================
    # Collect completed instance_ids
    # =====================
    BASE_OUTPUT_DIR = "llm_judge/result_llm_judge"
    completed_instance_ids = set()
    repo_path = BASE_OUTPUT_DIR  # Fix: Don't use os.path(...) here, it's redundant
    if os.path.isdir(repo_path):
        for fname in os.listdir(repo_path):
            if fname.endswith("_selected_similarbug.json"):
                instance_id = fname.replace("_selected_similarbug.json", "")
                completed_instance_ids.add(instance_id)

    swe_bench_lite = load_dataset("princeton-nlp/SWE-bench_lite", split="test")
    for instance in tqdm(swe_bench_lite, desc="Processing SWE-bench-lite"):
        owner, repo = instance["repo"].split("/")
        if repo != 'django':
            continue
        
        instance_id = instance["instance_id"]
        print(f" ------ Processing instance: {instance_id} ------ ")
        
        # Extract key information from current instance
        repo_full = instance["repo"]  # e.g., "astropy/astropy"
        created_at = instance.get("created_at", "")  # Creation time
        problem_statement = instance["problem_statement"].strip()  # Problem description
        patch = instance.get("patch", "")  # Patch information
        
        # Skip completed items
        if instance_id in completed_instance_ids:
            print(f"Skipping completed instance: {instance_id}")
            continue
        # Get current bug report content
        current_bug = problem_statement
        # Read JSON file
        with open(f"exemplar_mining/candidates/{repo}/{instance_id}_similarbugs_v1.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract top 5 candidate bug information
        candidates = data.get("similar_bug_items", [])[:5]
        candidate_bug = dict()
        num = 0 
        for item in candidates:
            num += 1
            bug_report, pr, file_paths = extract_candidate_info(item)
            if num_tokens(bug_report) >MAX_TOKENS_PER_REPORT:
                bug_report = truncate_by_token_limit(bug_report, MAX_TOKENS_PER_REPORT)
            if num_tokens(pr) > MAX_TOKENS_PER_REPORT:
                pr = truncate_by_token_limit(pr, MAX_TOKENS_PER_REPORT)
            candidate_bug[f"Candidates_{num}"] = {"bug_report": bug_report, "pr": pr, "file_paths": file_paths}

        # ✅ Automatically pad missing candidate items
        candidate_bug = pad_candidate_bug(candidate_bug)

        # # Fill in prompt template with content
        filled_prompt = prompt_template.format(
            current_bug_report=current_bug,
            bug_report_a=candidate_bug["Candidates_1"]["bug_report"], pr_a=candidate_bug["Candidates_1"]["pr"], file_paths_a=candidate_bug["Candidates_1"]["file_paths"],
            bug_report_b=candidate_bug["Candidates_2"]["bug_report"], pr_b=candidate_bug["Candidates_2"]["pr"], file_paths_b=candidate_bug["Candidates_3"]["file_paths"],
            bug_report_c=candidate_bug["Candidates_3"]["bug_report"], pr_c=candidate_bug["Candidates_3"]["pr"], file_paths_c=candidate_bug["Candidates_3"]["file_paths"],
            bug_report_d=candidate_bug["Candidates_4"]["bug_report"], pr_d=candidate_bug["Candidates_4"]["pr"], file_paths_d=candidate_bug["Candidates_4"]["file_paths"],
            bug_report_e=candidate_bug["Candidates_5"]["bug_report"], pr_e=candidate_bug["Candidates_5"]["pr"], file_paths_e=candidate_bug["Candidates_5"]["file_paths"]
        )

        # print(filled_prompt)
            # Call GPT-4o mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": filled_prompt}]
            # temperature=0.2,
            # max_tokens=800
        )
       
        response_str = response.choices[0].message.content
        
        final_choice, justification = extract_final_choice_and_justification(response_str)
        print("final choice:",final_choice,"\n","justification:",justification)

        choice_to_index = {
        "CANDIDATE_A": 0,
        "CANDIDATE_B": 1,
        "CANDIDATE_C": 2,
        "CANDIDATE_D": 3,
        "CANDIDATE_E": 4
        }

        # Get the selected candidate index
        selected_index = choice_to_index.get(final_choice, None)

        if selected_index is not None and selected_index < len(candidates):
            selected_item = candidates[selected_index]

            # Build output structure (following result_llm_judge format)
            output_data = {
                "instance_id": instance_id,
                "repo": repo_full,
                "created_at": created_at,
                "problem_statement": problem_statement,
                "patch": patch,
                "Selected_candidate": selected_item,
                "Justification": justification
            }

            # print(output_data)
            # Save to new JSON file
            output_file = f"output/result_llm_judge/{instance_id}_selected_similarbug.json"
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            print(f"Selected candidate saved to {output_file}")
        else:
            print("*******Invalid final choice or index out of range.*******"+"\n"+"*******Invalid final choice or index out of range.*******"+"\n"+"*******Invalid final choice or index out of range.*******")
        
