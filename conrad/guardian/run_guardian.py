#!/usr/bin/env python3
"""
Process summaries files and evaluate usefulness of similar issues using judge.py
"""

import os
import json
import glob
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from guardian import judge_batch, judge_batch_with_reflection

# Load environment variables from .env (if present)
load_dotenv()

def load_summaries_file(file_path):
    """Load summaries JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None

def prepare_candidates_from_summaries(summaries_data):
    """
    Convert summaries data to candidates format required by judge
    
    Args:
        summaries_data: Data containing original_problem and similar_issues_summaries
    
    Returns:
        candidates: List[Dict] format suitable for judge_batch
    """
    candidates = []
    
    similar_issues = summaries_data.get('similar_issues_summaries', [])
    
    for i, issue_data in enumerate(similar_issues, 1):
        similar_issue = issue_data.get('similar_issue', {})
        summary = issue_data.get('summary', '')
        
        # Build candidate
        candidate = {
            'id': f"similar_{similar_issue.get('issue_id', i)}",
            'summary': summary,  # Fixed field name to summary
            'patch': '',  # Similar issues don't have patch info, use empty string
            'similar_issue': similar_issue,  # Add similar_issue data for judge.py access
            'prompt_used': issue_data.get('prompt_used', '')  # Add prompt_used to extract changes info
        }
        candidates.append(candidate)
    
    return candidates

def create_llm_function(api_key, model_name="gpt-4o", base_url=None):
    """
    Create LLM function
    
    Args:
        api_key: API key
        model_name: Model name, supports "gpt-4o", "gpt-5", "deepseek-v3", etc.
        base_url: API base URL for third-party models (e.g., DeepSeek)
    
    Returns:
        LLM function
    """
    # Configure client based on model
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    # Set temperature based on model type
    # gpt-5 only supports default temperature=1
    if model_name == "gpt-5":
        temperature = 1
    else:
        temperature = 0.5
    
    def llm_call(payload):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=payload["messages"],
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed (model={model_name}): {e}")
            return '{"candidates": []}'
    
    return llm_call

def create_openai_llm(api_key):
    """Create OpenAI LLM function (for backward compatibility)"""
    return create_llm_function(api_key, model_name="gpt-4o")

def process_single_summaries_file(file_path, llm_func, output_dir):
    """Process single summaries file"""
    # Load summaries data
    summaries_data = load_summaries_file(file_path)
    if not summaries_data:
        return False
    
    # Detect JSON format and extract information
    if 'Selected_candidate' in summaries_data:
        # New format: result_llm_judge/*.json
        print(f"  Detected new format JSON (Selected_candidate)")
        
        # Current issue info is at top level
        current_issue_summary = summaries_data.get('problem_statement', '')
        current_patch = summaries_data.get('patch', '')
        
        if not current_issue_summary:
            print(f"  Warning: problem_statement not found in file {file_path}")
            return False
        
        # Build single candidate
        selected = summaries_data.get('Selected_candidate', {})
        issue_title = selected.get('issue_title', '')
        issue_body = selected.get('issue_body', '')
        pr_title = selected.get('pr_title', '')
        pr_number = selected.get('pr_number', 'unknown')
        
        candidates = [{
            'id': f"pr_{pr_number}",
            'summary': f"PR: {pr_title}\n\nIssue: {issue_title}\n\n{issue_body}",
            'patch': '',  # In new format patch is in changes, use empty string here
            'justification': summaries_data.get('Justification', ''),
            'similar_issue': {  # Add similar_issue field for judge.py usage
                'issue_title': issue_title,
                'issue_body': issue_body,
                'pr_number': pr_number,
                'pr_title': pr_title
            }
        }]
        
        # New format for original_problem construction
        original_problem = {
            'problem_statement': current_issue_summary,
            'patch': current_patch
        }
        
    else:
        # Old format: summaries/*.json
        print(f"  Detected old format JSON (original_problem)")
        original_problem = summaries_data.get('original_problem', {})
        current_issue_summary = original_problem.get('problem_statement', '')
        current_patch = original_problem.get('patch', '')
        
        if not current_issue_summary:
            print(f"  Warning: problem_statement not found in file {file_path}")
            return False
        
        # Prepare candidates
        candidates = prepare_candidates_from_summaries(summaries_data)
    
    if not candidates:
        print(f"  Warning: No candidates found in file {file_path}")
        return False
    
    # Evaluate candidates (new format has 1, old format takes 3rd)
    if 'Selected_candidate' in summaries_data:
        top1_candidates = candidates[:1]  # New format: take first (only one)
    else:
        top1_candidates = candidates[2:3]  # Old format: take third
    
    print(f"  Evaluating candidates (total {len(top1_candidates)})...")
    
    # Call judge
    try:
        # Build prompt first for saving
        from guardian import build_batch_prompt
        prompt_data = build_batch_prompt(
            current_issue_summary=current_issue_summary,
            current_patch=current_patch,
            candidates=top1_candidates
        )
        
        # Use judgment with reflection (can be controlled via parameter)
        use_reflection = True  # Set to True to enable self reflection
        
        if use_reflection:
            print(f"  Using Self Reflection mode...")
            full_judgment_result = judge_batch_with_reflection(
                llm=llm_func,
                current_issue_summary=current_issue_summary,
                current_patch=current_patch,
                candidates=top1_candidates,
                use_reflection=True
            )
            judgment_result = full_judgment_result["final_judgment"]
        else:
            full_judgment_result = None
            judgment_result = judge_batch(
                llm=llm_func,
                current_issue_summary=current_issue_summary,
                current_patch=current_patch,
                candidates=top1_candidates
            )
        
        # Build complete result
        result = {
            'original_problem': original_problem,
            'candidates_evaluated': len(top1_candidates),
            'judgment_result': judgment_result,
            'prompt_used': prompt_data,  # Save the prompt used
        }
        
        # Save raw_summaries (if old format)
        if 'similar_issues_summaries' in summaries_data:
            result['raw_summaries'] = summaries_data['similar_issues_summaries']
        elif 'Selected_candidate' in summaries_data:
            # New format saves Selected_candidate and Justification
            result['selected_candidate'] = summaries_data['Selected_candidate']
            result['justification'] = summaries_data.get('Justification', '')
        
        # If reflection was used, save complete reflection info
        if use_reflection and full_judgment_result:
            result['reflection_info'] = {
                'initial_judgment': full_judgment_result.get('initial_judgment'),
                'reflection_judgment': full_judgment_result.get('reflection_judgment'),
                'reflection_used': True
            }
        
        # Save result
        # Handle two filename formats
        filename = os.path.basename(file_path)
        if '_selected_similarbug.json' in filename:
            # New format: astropy__astropy-6938_selected_similarbug.json -> astropy__astropy-6938_judged.json
            output_filename = filename.replace('_selected_similarbug.json', '_judged.json')
        else:
            # Old format: xxx_summaries.json -> xxx_judged.json
            output_filename = filename.replace('_summaries.json', '_judged.json')
        
        # Determine save path based on judgment result
        # Check if first candidate is Useful
        first_candidate_decision = judgment_result['candidates'][0]['decision'] if judgment_result.get('candidates') else 'Not useful'
        
        if first_candidate_decision == 'Useful':
            # Save to accept subdirectory
            final_output_dir = os.path.join(output_dir, 'accept')
        else:
            # Save to discard subdirectory (includes Harmful and Not useful)
            final_output_dir = os.path.join(output_dir, 'discard')
        
        # Create target directory
        os.makedirs(final_output_dir, exist_ok=True)
        
        output_path = os.path.join(final_output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"  Judgment result saved to: {output_path}")
        
        # Print detailed statistics
        useful_count = sum(1 for c in judgment_result['candidates'] 
                          if c['decision'] == 'Useful')
        harmful_count = sum(1 for c in judgment_result['candidates'] 
                           if c['decision'] == 'Harmful')
        not_useful_count = sum(1 for c in judgment_result['candidates'] 
                              if c['decision'] == 'Not useful')
        
        print(f"  Result: {useful_count} Useful, {harmful_count} Harmful, {not_useful_count} Not useful (total {len(top1_candidates)})")
        
        return True
        
    except Exception as e:
        print(f"  Processing failed: {e}")
        return False

def process_all_summaries(summaries_dir="summaries", output_dir="judgments", api_key=None, instance_ids=None, 
                          model_name="gpt-4o", base_url=None):
    """
    Batch process all summaries files
    
    Args:
        summaries_dir: Path to summaries folder
        output_dir: Output directory
        api_key: API key
        instance_ids: List of instance IDs to process, if None process all files
        model_name: Model name, e.g., "gpt-4o", "deepseek-chat", "deepseek-reasoner"
        base_url: API base URL for third-party models
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LLM function
    llm_func = create_llm_function(api_key, model_name=model_name, base_url=base_url)
    print(f"Using model: {model_name}")
    if base_url:
        print(f"API address: {base_url}")
    
    # Find all summaries files
    # Support two filename formats:
    # 1. Old format: *_summaries.json
    # 2. New format: *_selected_similarbug.json
    pattern1 = os.path.join(summaries_dir, "*_summaries.json")
    pattern2 = os.path.join(summaries_dir, "*_selected_similarbug.json")
    all_summaries_files = glob.glob(pattern1) + glob.glob(pattern2)
    
    if not all_summaries_files:
        print(f"No summaries files found in {summaries_dir}")
        print(f"  Searched patterns: *_summaries.json or *_selected_similarbug.json")
        return
    
    # Filter files if instance_ids specified
    if instance_ids:
        filtered_files = []
        for file_path in all_summaries_files:
            # Extract instance_id from filename
            filename = os.path.basename(file_path)
            
            # Handle two file formats
            if '_selected_similarbug.json' in filename:
                # New format: astropy__astropy-6938_selected_similarbug.json
                instance_id = filename.replace('_selected_similarbug.json', '')
            else:
                # Old format: django_django__django-11001_similarbugs_v1_summaries.json
                base_name = filename.replace('_summaries.json', '')
                parts = base_name.split('_similarbugs_v1')[0]
                
                # Handle duplicate project names
                if '__' in parts:
                    instance_id = parts[parts.find('__')-len(parts.split('__')[0].split('_')[-1]):]
                else:
                    instance_id = parts
                
            if instance_id in instance_ids:
                filtered_files.append(file_path)
        
        summaries_files = filtered_files
        print(f"Processing specified {len(instance_ids)} instance IDs, found {len(summaries_files)} matching files")
        
        # Print missing instance IDs
        found_ids = set()
        for file_path in summaries_files:
            filename = os.path.basename(file_path)
            
            # Extract instance_id based on file format
            if '_selected_similarbug.json' in filename:
                instance_id = filename.replace('_selected_similarbug.json', '')
            else:
                base_name = filename.replace('_summaries.json', '')
                parts = base_name.split('_similarbugs_v1')[0]
                if '__' in parts:
                    instance_id = parts[parts.find('__')-len(parts.split('__')[0].split('_')[-1]):]
                else:
                    instance_id = parts
            found_ids.add(instance_id)

        missing_ids = set(instance_ids) - found_ids
        if missing_ids:
            print(f"Files not found for the following instance IDs: {list(missing_ids)[:10]}...")  # Only show first 10
    else:
        summaries_files = all_summaries_files
        print(f"Found {len(summaries_files)} summaries files")
    
    if not summaries_files:
        print("No files to process")
        return
    
    success_count = 0
    # Statistics for idx=1 candidates
    idx1_stats = {"Useful": 0, "Not useful": 0, "Harmful": 0, "Total": 0}
    
    for file_path in summaries_files:
        print(f"\\nProcessing file: {file_path}")
        if process_single_summaries_file(file_path, llm_func, output_dir):
            success_count += 1
            
            # Collect stats for idx=1 candidate in this file
            try:
                # Build output file path
                filename = os.path.basename(file_path)
                if '_selected_similarbug.json' in filename:
                    output_filename = filename.replace('_selected_similarbug.json', '_judged.json')
                else:
                    base_filename = os.path.splitext(filename)[0]
                    output_filename = base_filename.replace('_summaries', '_judged') + '.json'
                
                # Check both possible locations (accept and discard subdirectories)
                output_path_accept = os.path.join(output_dir, 'accept', output_filename)
                output_path_discard = os.path.join(output_dir, 'discard', output_filename)
                
                # Try both paths
                output_path = None
                if os.path.exists(output_path_accept):
                    output_path = output_path_accept
                elif os.path.exists(output_path_discard):
                    output_path = output_path_discard
                
                # Read judgment result file
                if output_path and os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        judge_data = json.load(f)
                    
                    # Find idx=1 candidate
                    candidates = judge_data.get('judgment_result', {}).get('candidates', [])
                    for candidate in candidates:
                        if candidate.get('idx') == 1:
                            decision = candidate.get('decision', 'Unknown')
                            if decision in idx1_stats:
                                idx1_stats[decision] += 1
                            idx1_stats["Total"] += 1
                            break  # Only count first idx=1 candidate
            except Exception as e:
                print(f"Error collecting stats for file {file_path}: {e}")
    
    print(f"\\nProcessing complete: {success_count}/{len(summaries_files)} files processed successfully")
    
    # Output idx=1 candidate statistics
    print(f"\\n📊 idx=1 Candidate Judgment Statistics:")
    print(f"  Total: {idx1_stats['Total']}")
    useful_pct = idx1_stats['Useful']/max(idx1_stats['Total'], 1)*100
    not_useful_pct = idx1_stats['Not useful']/max(idx1_stats['Total'], 1)*100
    harmful_pct = idx1_stats['Harmful']/max(idx1_stats['Total'], 1)*100
    print(f"  Useful: {idx1_stats['Useful']} ({useful_pct:.1f}%)")
    print(f"  Not useful: {idx1_stats['Not useful']} ({not_useful_pct:.1f}%)")
    print(f"  Harmful: {idx1_stats['Harmful']} ({harmful_pct:.1f}%)")

if __name__ == "__main__":
    # Configure instance IDs to process (if None, process all files)
    # Option 1: Specify instance IDs list
    # target_instance_ids = [
    #     "astropy__astropy-6938",
    #     "django__django-10914"
    # ]
    
    # Option 2: Set to None to process all files in the directory
    target_instance_ids = None
    
    # ========== Model Selection ==========
    print("\n" + "="*60)
    print("Please select the model to use:")
    print("="*60)
    print("1. gpt-4o (OpenAI)")
    print("2. gpt-5 (OpenAI)")
    print("3. deepseek-v3 (DeepSeek)")
    print("4. Custom model")
    print("="*60)

    model_choice = input("Enter option (1-4, default is 1): ").strip() or "1"

    # Model configurations
    # Note: DeepSeek V3's common API model name might be `deepseek-chat`; displayed as deepseek-v3 externally.
    model_configs = {
        "1": {
            "name": "gpt-4o",
            "base_url": None,
            "api_key_prompt": "Enter OpenAI API Key: "
        },
        "2": {
            "name": "gpt-5",
            "base_url": None,
            "api_key_prompt": "Enter OpenAI API Key: "
        },
        "3": {
            "name": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key_prompt": "Enter DeepSeek API Key: "
        }
    }
    
    if model_choice == "4":
        model_name = input("Enter model name: ").strip()
        base_url = input("Enter API base URL (leave empty for OpenAI): ").strip() or None
        api_key_prompt = "Enter API Key: "
    elif model_choice in model_configs:
        config = model_configs[model_choice]
        model_name = config["name"]
        base_url = config["base_url"]
        api_key_prompt = config["api_key_prompt"]
    else:
        print("Invalid option, using default model gpt-4o")
        model_name = "gpt-4o"
        base_url = None
        api_key_prompt = "Enter OpenAI API Key: "
    
    # Get API key (prioritize environment variable to avoid hardcoding key in code)
    if base_url:
        # DeepSeek recommends DEEPSEEK_API_KEY; can also use OPENAI_API_KEY for compatibility
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("INFO: DEEPSEEK_API_KEY or OPENAI_API_KEY not found in environment")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("INFO: OPENAI_API_KEY not found in environment")

    if not api_key:
        api_key = input(api_key_prompt).strip()
        if not api_key:
            print("Error: API Key must be provided")
            exit(1)
    
    display_model_name = "deepseek-v3" if (base_url and model_name == "deepseek-chat") else model_name
    print(f"\nSelected model: {display_model_name}")
    if base_url:
        print(f"API address: {base_url}")
    print("="*60 + "\n")
    
    if target_instance_ids:
        print(f"Starting to process specified {len(target_instance_ids)} instance IDs...")
    else:
        print("Starting to process all summaries files for judgment...")
    
    # Process summaries files
    # Use new result_llm_judge folder path
    process_all_summaries(
        summaries_dir="output/result_llm_judge",
        output_dir="output/guardian_results",
        api_key=api_key,
        instance_ids=target_instance_ids,  # Pass instance IDs list
        model_name=model_name,
        base_url=base_url
    )
