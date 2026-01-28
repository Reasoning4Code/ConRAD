import openai
import json
from typing import List, Dict, Any, Union
import os
from dotenv import load_dotenv

from get_repo_structure import (
    get_project_structure_from_scratch,
    filter_out_test_files,
    filter_none_python,
    show_project_structure,
)
from get_content import (
    extract_function_from_patch,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    localize_function_from_compressed_files,
)
from get_patch import (
    get_diff_from_pr,
    extract_file_from_patch,
    extract_function_from_patch,
    extract_line_from_patch
)
from test import (
    parse_think_and_answer,
    extract_code_blocks,
    extract_locs_for_files
)

# API key should not be hard-coded. Read from environment for safe publishing.
import sys

# Load environment variables from .env (if present)
load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
if not API_KEY:
    sys.stderr.write("WARNING: OPENAI_API_KEY environment variable not set. Some features requiring the OpenAI API will fail.\n")

# Repository mapping
repo_map = {
    "django": "django/django",
    "sphinx-doc": "sphinx-doc/sphinx",
    "pytest-dev": "pytest-dev/pytest",
    "scikit-learn": "scikit-learn/scikit-learn",
    "sympy": "sympy/sympy",
    "pallets": "pallets/flask",
    "pylint-dev": "pylint-dev/pylint",
    "matplotlib": "matplotlib/matplotlib",
    "astropy": "astropy/astropy",
    "psf": "psf/requests"
}

# Prompt templates with ground truth hints
generate_prompt_with_gt_hint = {
    1: """
You are an expert software engineer tasked with locating the most relevant file for fixing a bug.

You are given:
- A GitHub Issue Description.
- A snapshot of the project Project Directory Structure (program file paths only).
- The actual file that was modified to fix this GitHub Issue (ground truth).

Your task is to **reason step-by-step** how to narrow down the possible files **starting from the entire directory structure**, and explain why the ground truth file is the most appropriate for this bug fix.

Avoid vague jumps. Instead, explain your reasoning like a detective narrowing down suspects.

---

<GitHub Issue Description>
{problem}
</GitHub Issue Description>

<Project Directory Structure>
{repository_structure}
</Project Directory Structure>

<Ground Truth Modified Files>
{gt_files}
</Ground Truth Modified Files>

---
Step-by-step reasoning:
1. First, I look at the bug report to extract key terms, affected modules, and clues.
2. Then, I scan the directory structure to find files or directories whose names or paths semantically relate to those clues.
3. Among those, I consider the functionality suggested by the bug (e.g., GPU, utils, tensors).
4. Based on likely responsibilities and location of logic, I narrow it down further.
5. I find that the files "{gt_files}" are the best match because...

Now please provide the full step by step reasoning, following the output format:
- Start your answer with "Step 1:" (no preamble).
- Write one step per line, numbered consecutively as "Step 1:", "Step 2:", ...


And provide the full path and return at most 5 files. The returned files should be separated by new lines ordered by most to least important.
For example:
```
file1.py
file2.py
```
Return the location(s) wrapped with ```
Your reasoning should start with "### Thinking:", and your answer should start with "### Answer:".
""",

    2: """
You are an expert software engineer tasked with locating the most relevant locations for fixing a bug.

Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

You are given:
- A GitHub Problem Description.
- A snapshot of the Skeleton of Relevant Files.
- The actual locations were modified to fix this bug (ground truth), which can be the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

Your task is to **reason step-by-step** how to narrow down the possible locations**starting from the entire Skeleton of Relevant Files**, and explain why the ground truth locations are the most appropriate for this bug fix.

Avoid vague jumps. Instead, explain your reasoning like a detective narrowing down suspects.

---

## GitHub Problem Description:
{problem}

## Skeleton of Relevant Files:
{file_skeleton}

## Ground Truth Locations:
# The following entities have been identified as **very likely candidates** for being involved in the issue. Beyond your standard reasoning, please give special attention to the following entities, as they are very likely to be relevant to the issue.
{gt_related_elements}

---

Step-by-step reasoning:
1. First, I look at the bug report to extract key terms, affected modules, and clues.
2. Then, I scan the Skeleton of Relevant Files to find classes, functions, or directories whose names or paths semantically relate to those clues.
3. Among those, I consider the functionality suggested by the bug.
4. Based on likely responsibilities and locations of logic, I narrow it down further.
5. I find that the locations "{gt_related_elements}" are the best match because...

Now explain your full reasoning step-by-step, Now please provide the full step by step reasoning, following the output format:
- Start your answer with "Step 1:" (no preamble).
- Write one step per line, numbered consecutively as "Step 1:", "Step 2:", ...
- Do not include any conversational text (e.g., "Sure", "Here is", etc.).

And please provide the complete set of locations as either a class name, a function name, or a variable name. Note that if you include a class, you do not need to list its specific methods. You can include either the entire class or don't include the class name and instead include specificmethods in the class.
For example:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```
Return the location(s) wrapped with ```
Your reasoning should start with "### Thinking:", and your answer should start with "### Answer:".
""",

    3: """Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}
###

### Additional Contextual Clues:
The following edit location(s) have been identified as **very likely candidates** for being involved in the issue. Beyond your standard reasoning, please give special attention to the following edit location(s), as they are very likely to be relevant to the issue.
{gt_edit_locs}

---
Step-by-step reasoning:
1. First, I review the bug report to identify the nature of the change — such as logic updates, condition insertions, or bug fixes tied to specific scenarios.
2. Then, I examine the function(s)/class(es)/variable(s) identified in the previous step, and read their full code contents to understand where the faulty or missing logic lies.
3. I look for:
   - The place where the bug-triggering logic occurs;
   - The control flow point where a fix (e.g., an if-check or assignment) could be inserted;
   - Specific lines that should be edited or augmented to address the issue.
4. I match this location with line numbers and ensure they align with the actual faulty behavior or the fix to be added.
5. I include only the minimal set of relevant line numbers, and if needed, also mention the function or class context for clarity.


Now please provide the full step by step reasoning, following the output format:
- Start your answer with "Step 1:" (no preamble).
- Write one step per line, numbered consecutively as "Step 1:", "Step 2:", ...
- Do not include any conversational text (e.g., "Sure", "Here is", etc.).

And please provide the class name, function or method name, or the exact line numbers that need to be edited.
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```
Return the location(s) wrapped with ```
Please think step by step before returning the location(s).
Your reasoning should start with "### Thinking:", and your answer should start with "### Answer:".
""",

    4: """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
```
{file_contents}
```
--- END FILE ---


### Additional Contextual Clues:
The following *SEARCH/REPLACE* edit have been identified as **the ground truth** for being involved in the issue. Beyond your standard reasoning, please give special attention to the following *SEARCH/REPLACE* edit.
{search_replace}

---

Step-by-step reasoning:
1. I begin by analyzing the bug report to identify the expected behavior and the faulty or missing logic.
2. Then, I locate the relevant file and function or class by reading through the provided code segments.
3. I identify the exact lines that cause the bug or where the fix needs to be inserted (e.g., a missing condition, incorrect return, or broken logic).
4. I select a contiguous code block from the current code that I would like to change — this is the *SEARCH* block.
5. I then write the correct version of the code as the *REPLACE* block, ensuring all indentation and spacing is preserved exactly.
6. I format the result in the required SEARCH/REPLACE format and wrap it in a code block.

Now please provide the full step by step reasoning, following the output format:
- Start your answer with "Step 1:" (no preamble).
- Write one step per line, numbered consecutively as "Step 1:", "Step 2:", ...

And then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line 'print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```
"""
}


def run_model(prompt: str, model: str = "gpt-5", max_output_tokens: int = 30000) -> str:
    """
    Use Responses API (recommended for GPT-5 / reasoning models)
    """
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in the environment or see .env.example.")
    client = openai.OpenAI(api_key=API_KEY)
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return ""


def should_skip_file(filename):
    """Determine whether to skip a file"""
    filename_lower = filename.lower()
    
    # 1. Only process Python files
    if not filename_lower.endswith('.py'):
        return True
    
    # 2. Filter test files - check each part of the path
    path_parts = filename_lower.replace('\\', '/').split('/')
    
    for part in path_parts:
        # Check if any path part starts with 'test'
        if part.startswith('test'):
            return True
    return False


def transform_diff_to_search_replace_single_file(diff, prompt):
    """
    Convert a single file's diff to SEARCH/REPLACE format
    """
    search_replace_content = ""
    search_lines = []
    replace_lines = []
    diff_list = diff.strip().splitlines()
    begin_index = 0
    filename = ""
    
    try:
        while not diff_list[begin_index].startswith("+++ b/"):
            begin_index += 1
    except Exception as e:
        print("Error while parse diff:", e)
        return None
        
    filename = diff_list[begin_index][6:]
    
    # Skip non-Python files and test files
    if should_skip_file(filename):
        print(f"Skipping file: {filename} (not Python or is test file)")
        return None
    
    begin_index += 2
    for line in diff_list[begin_index:]:
        if line.startswith("@@") and search_lines:
            search_content = '\n'.join(search_lines)
            replace_content = '\n'.join(replace_lines)
            if search_content in prompt:
                search_replace_content += f'''<<<<<<< SEARCH
{search_content}
=======
{replace_content}
>>>>>>> REPLACE
'''
            search_lines = []
            replace_lines = []
            continue
        if line.startswith("+"):
            replace_lines.append(line[1:])
        elif line.startswith("-"):
            search_lines.append(line[1:])
        else:
            search_lines.append(line[1:])
            replace_lines.append(line[1:])
    
    search_content = '\n'.join(search_lines)
    replace_content = '\n'.join(replace_lines)
    if search_content in prompt:
        search_replace_content += f'''<<<<<<< SEARCH
{search_content}
=======
{replace_content}
>>>>>>> REPLACE
'''
    
    if not search_replace_content:
        return None
    
    search_replace_content = f'''<patch>
```python
### {filename}
{search_replace_content}
```
</patch>
'''
    return search_replace_content


def transform_diff_to_search_replace(diff, prompt):
    """
    Convert patch file (diff) content to search and replace format string.
    Only process Python files, skip test files.
    """
    diff_files = diff.split("diff --git")
    search_replace_content_list = []
    skipped_files = []
    
    for diff_file in diff_files[1:]:
        search_replace_content = transform_diff_to_search_replace_single_file("diff --git" + diff_file, prompt)
        if search_replace_content:
            search_replace_content_list.append(search_replace_content.split("<patch>\n```python\n")[1].split("\n```\n</patch>")[0])
        else:
            lines = diff_file.strip().splitlines()
            for line in lines:
                if line.startswith("+++ b/"):
                    skipped_files.append(line[6:])
                    break
    
    if skipped_files:
        print(f"Skipped files: {skipped_files}")
        
    if search_replace_content_list:
        search_replace_content = "<patch>\n```python\n" + "\n\n".join(search_replace_content_list) + "\n```\n</patch>"
        return search_replace_content
    return None


def filter_diff_by_file_type(diff):

    diff_files = diff.split("diff --git")
    filtered_diff_list = []
    skipped_files = []
    
    for diff_file in diff_files[1:]:  
        full_diff_file = "diff --git" + diff_file
        

        lines = diff_file.strip().splitlines()
        filename = ""
        
        try:
            for line in lines:
                if line.startswith("+++ b/"):
                    filename = line[6:]
                    break
        except Exception as e:
            print(f"Error while parsing diff: {e}")
            continue
        
        if filename:
            if should_skip_file(filename):
                print(f"Skipping file: {filename} (not Python or is test file)")
                skipped_files.append(filename)
            else:
                filtered_diff_list.append(full_diff_file)
        
    if skipped_files:
        print(f"Skipped files: {skipped_files}")
    
    if filtered_diff_list:
        filtered_diff = '\n'.join(filtered_diff_list)
        return filtered_diff
    
    return None


def write_json(data: Union[Dict, List, Any], file_path: str, indent: int = 2, ensure_ascii: bool = False) -> bool:

    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        print(f"JSON data successfully written to: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error occurred while writing to the JSON file: {e}")
        return False


if __name__ == "__main__":
    instance_txt_path = 'backward_distillation/gpt5_10.txt'
    instance_ids = []
    with open(instance_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            instance_id = line.strip()
            if instance_id:
                instance_ids.append(instance_id)

    for instance_id in instance_ids:
        print('Processing the instance:', instance_id)
        
        prefix = instance_id.split("__")[0]
        if prefix not in repo_map:
            print(f"Unknown prefix:{prefix}, skip {instance_id}")
            continue
        
        repo = repo_map[prefix]
        

        with open(f"backward_distillation/result_llm_judge/{instance_id}_selected_similarbug.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        

        issue_title = data["Selected_candidate"]["issue_title"]
        issue_body = data["Selected_candidate"]["issue_body"]
        PROBLEM = issue_title + "\n" + issue_body
        base_commit = data["Selected_candidate"]["base_commit"]
        pr_number = data["Selected_candidate"]["pr_number"]
        

        # Use GitHub token from environment for API calls (do not hardcode tokens)
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not github_token:
            print("WARNING: GITHUB_TOKEN not set in environment. GitHub API calls may fail.")
        tokens = [github_token] if github_token else []
        patch = get_diff_from_pr(repo=repo, pr_number=pr_number, tokens=tokens, max_retries=5)
        filepaths = extract_file_from_patch(patch)
        ground_functions_raw, found_related_locs = extract_function_from_patch(patch)
        ground_lines = extract_line_from_patch(patch)
        
        print('Patch:', "\n", patch)
        print("Ground files:", filepaths)
        print("Ground function raw:", ground_functions_raw, '\nProcessed:', found_related_locs)
        print("Ground lines:", ground_lines)
        
        gt_files = filepaths
        gt_related_elements = found_related_locs
        gt_edit_locs = ground_lines
        edit_blocks = extract_code_blocks(gt_edit_locs)
        edit_locs = extract_locs_for_files(edit_blocks, gt_files, False)
        print("Edit locs:", edit_locs)
        
      
        d = get_project_structure_from_scratch(repo, base_commit, instance_id, "playground")
        structure = d["structure"]
        filter_none_python(structure)
        filter_out_test_files(structure)
        structure_processed = show_project_structure(structure)
        

        skeleton = localize_function_from_compressed_files(structure, filepaths)
        
        from repair.repair import construct_topn_file_context
        
        file_contents = dict()
        files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
        
        existing_gt_files = []
        for i, pred_file in enumerate(gt_files):
            content = None
            for file_content in files:
                if file_content[0] == pred_file:
                    content = "\n".join(file_content[1])
                    file_contents[pred_file] = content
                    existing_gt_files.append(pred_file)
                    break
            if content is None:
                print(f"Warning: file not found, skip {pred_file}")
        
        if not existing_gt_files:
            print("No existing files found, skipping topn_content construction")
            topn_content = ""
            file_loc_intervals = {}
        else:
            existing_edit_locs = {k: v for k, v in edit_locs.items() if k in existing_gt_files}
            
            topn_content, file_loc_intervals = construct_topn_file_context(
                existing_edit_locs,
                existing_gt_files,
                file_contents,
                structure,
                context_window=10,
                loc_interval=True,
                fine_grain_loc_only=False,
                add_space=False,
                no_line_number=True,
                sticky_scroll=False,
            )
        
        filtered_patch = filter_diff_by_file_type(patch)
        search_replace = transform_diff_to_search_replace(filtered_patch, topn_content)
        
        file_prompt = generate_prompt_with_gt_hint[1].format(
            problem=PROBLEM,
            repository_structure=structure_processed.strip(),
            gt_files=gt_files,
        )
        file_response = run_model(file_prompt)
        file_think, file_answer = parse_think_and_answer(file_response)

        func_prompt = generate_prompt_with_gt_hint[2].format(
            problem=PROBLEM,
            file_skeleton=skeleton.strip(),
            gt_related_elements=gt_related_elements,
        )
        func_response = run_model(func_prompt)
        func_think, func_answer = parse_think_and_answer(func_response)

        edit_prompt = generate_prompt_with_gt_hint[3].format(
            problem_statement=PROBLEM,
            file_contents=topn_content,
            gt_edit_locs=gt_edit_locs
        )
        edit_response = run_model(edit_prompt)
        edit_think, edit_answer = parse_think_and_answer(edit_response)

        patch_prompt = generate_prompt_with_gt_hint[4].format(
            problem=PROBLEM,
            file_contents=topn_content,
            search_replace=search_replace
        )
        patch_response = run_model(patch_prompt)
        patch_think, patch_answer = parse_think_and_answer(patch_response)

        data2json = {
            "instance_id": instance_id,
            "problem": PROBLEM,
            "repository_structure": structure_processed.strip(),
            "file_skeleton": skeleton.strip(),
            "file_content": topn_content,
            "gt_files": gt_files,
            "gt_related_elements": gt_related_elements,
            "gt_edit_locs": gt_edit_locs,
            "search_replace": search_replace,
            "patch": filtered_patch,
            "file_think": file_think,
            "func_think": func_think,
            "edit_think": edit_think,
            "patch_think": patch_think,
        }
        write_json(data2json, f"gpt5/reasoning/{instance_id}.json")
