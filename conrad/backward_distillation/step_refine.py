"""
Step Refine Program
Function: Read reasoning files, split by steps, generate 3 new versions for each step, and compare quality
"""
import openai
import json
import os
import re
from typing import List, Dict, Tuple, Any
from conrad.backward_distillation.backward_distillation import API_KEY, run_model

def parse_steps(reasoning_text: str) -> List[Dict[str, Any]]:
    """
    Parse reasoning text and extract each step
    
    Args:
        reasoning_text: Reasoning text containing steps
    
    Returns:
        List of steps, each step contains step number and content
    """
    # Match steps in "Step X:" format
    pattern = r'(Step\s+\d+:\s*.*?)(?=Step\s+\d+:|$)'
    matches = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
    
    steps = []
    for i, match in enumerate(matches, 1):
        step_text = match.strip()
        steps.append({
            'step_number': i,
            'content': step_text
        })
    
    return steps


def generate_step_variants(
    step_content: str,
    step_number: int,
    context_before: str,
    context_after: str,
    problem: str,
    stage: str,
    stage_context: Dict[str, str],
    num_variants: int = 3
) -> List[str]:
    """
    Generate multiple variant versions for a specific step
    
    Args:
        step_content: Original step content
        step_number: Step number
        context_before: Content of previous steps (context)
        context_after: Content of subsequent steps (context)
        problem: Problem description
        stage: Current stage (file_think, func_think, edit_think, patch_think)
        stage_context: Stage-specific context information
        num_variants: Number of variants to generate
    
    Returns:
        List of generated variants
    """
    
    # Define different prompt templates for different stages
    stage_prompts = {
        'file_think': f"""You are an expert at file localization for bug fixing tasks.

## Task Description:
You need to locate the most relevant files that need to be edited to fix a bug, based on the issue description and repository structure.

## GitHub Issue:
{problem}

## Repository Structure:
{stage_context.get('repository_structure', 'Not provided')}

## Context Before This Step:
{context_before if context_before else "This is the first step."}

## Current Step {step_number}:
{step_content}

## Context After This Step:
{context_after if context_after else "This is the last step."}

---

Please generate {num_variants} improved variants of Step {step_number} ONLY.
Each variant should:
1. Provide more specific reasoning about file locations based on the repository structure
2. Explain WHY certain files/directories are relevant to the issue
3. Use technical terminology related to the project structure
4. Maintain consistency with surrounding steps

Each variant should start with "Step {step_number}:" and focus on file localization reasoning.

Format your response as:
### Variant 1:
[your improved step here]

### Variant 2:
[your improved step here]

### Variant 3:
[your improved step here]
""",
        
        'func_think': f"""You are an expert at function/class localization for bug fixing tasks.

## Task Description:
You need to locate the specific functions, classes, or methods that need to be inspected or modified to fix a bug, based on the file skeleton.

## GitHub Issue:
{problem}

## File Skeleton (Classes and Functions):
{stage_context.get('file_skeleton', 'Not provided')}

## Context Before This Step:
{context_before if context_before else "This is the first step."}

## Current Step {step_number}:
{step_content}

## Context After This Step:
{context_after if context_after else "This is the last step."}

---

Please generate {num_variants} improved variants of Step {step_number} ONLY.
Each variant should:
1. Provide specific analysis of which functions/classes/methods are relevant
2. Explain the relationships between components
3. Reference the file skeleton structure
4. Maintain consistency with surrounding steps

Each variant should start with "Step {step_number}:" and focus on function/class localization reasoning.

Format your response as:
### Variant 1:
[your improved step here]

### Variant 2:
[your improved step here]

### Variant 3:
[your improved step here]
""",
        
        'edit_think': f"""You are an expert at identifying exact edit locations for bug fixing tasks.

## Task Description:
You need to identify the exact lines or code blocks that need to be edited to fix a bug, based on the file contents.

## GitHub Issue:
{problem}

## File Contents:
{stage_context.get('file_content', 'Not provided')}

## Context Before This Step:
{context_before if context_before else "This is the first step."}

## Current Step {step_number}:
{step_content}

## Context After This Step:
{context_after if context_after else "This is the last step."}

---

Please generate {num_variants} improved variants of Step {step_number} ONLY.
Each variant should:
1. Be more precise about which lines or code blocks need editing
2. Explain WHY those specific locations are the right places to fix
3. Reference the actual code structure
4. Maintain consistency with surrounding steps

Each variant should start with "Step {step_number}:" and focus on identifying exact edit locations.

Format your response as:
### Variant 1:
[your improved step here]

### Variant 2:
[your improved step here]

### Variant 3:
[your improved step here]
""",
        
        'patch_think': f"""You are an expert at generating code patches for bug fixing tasks.

## Task Description:
You need to generate the exact SEARCH/REPLACE patch to fix a bug, based on the file contents.

## GitHub Issue:
{problem}

## File Contents:
{stage_context.get('file_content', 'Not provided')}

## Context Before This Step:
{context_before if context_before else "This is the first step."}

## Current Step {step_number}:
{step_content}

## Context After This Step:
{context_after if context_after else "This is the last step."}

---

Please generate {num_variants} improved variants of Step {step_number} ONLY.
Each variant should:
1. Provide clearer reasoning about the code changes needed
2. Explain the logic behind the fix
3. Reference specific code patterns or structures
4. Maintain consistency with surrounding steps

Each variant should start with "Step {step_number}:" and focus on patch generation reasoning.

Format your response as:
### Variant 1:
[your improved step here]

### Variant 2:
[your improved step here]

### Variant 3:
[your improved step here]
"""
    }
    
    # Select prompt for the corresponding stage
    prompt = stage_prompts.get(stage, f"""You are an expert at refining reasoning steps for software engineering tasks.

Given the following problem and reasoning step, please generate {num_variants} improved variants of this specific step.

## Problem Description:
{problem}

## Context Before This Step:
{context_before if context_before else "This is the first step."}

## Current Step {step_number}:
{step_content}

## Context After This Step:
{context_after if context_after else "This is the last step."}

---

Please generate {num_variants} improved variants of Step {step_number} ONLY.
Each variant should start with "Step {step_number}:" and maintain consistency with the before/after context.

Format your response as:
### Variant 1:
[your improved step here]

### Variant 2:
[your improved step here]

### Variant 3:
[your improved step here]
""")
    
    response = run_model(prompt, max_output_tokens=4000)
    
    # Parse variants
    variants = []
    variant_pattern = r'### Variant \d+:\s*(.*?)(?=### Variant \d+:|$)'
    variant_matches = re.findall(variant_pattern, response, re.DOTALL)
    
    for match in variant_matches:
        variants.append(match.strip())
    
    return variants


def compare_step_variants(
    original_step: str,
    variants: List[str],
    step_number: int,
    problem: str,
    ground_truth: str
) -> Dict[str, Any]:
    """
    Compare original step and all variants, select the best version
    
    Args:
        original_step: Original step
        variants: List of generated variants
        step_number: Step number
        problem: Problem description
        ground_truth: Ground truth (used for evaluation)
    
    Returns:
        Evaluation results, including best version and scores
    """
    all_candidates = [original_step] + variants
    
    prompt = f"""You are an expert evaluator of reasoning quality for software engineering tasks.

## Problem Description:
{problem}

## Ground Truth outcome:
{ground_truth}

## Original Step {step_number}:
{original_step}

## Variant 1:
{variants[0] if len(variants) > 0 else "N/A"}

## Variant 2:
{variants[1] if len(variants) > 1 else "N/A"}

## Variant 3:
{variants[2] if len(variants) > 2 else "N/A"}

---

Please evaluate each version (Original, Variant 1, Variant 2, Variant 3) based on:
1. Consistency with the surrounding reasoning plan context.
2. Compatibility with the verified outcomes Y, especially the final patch y_patch.
3. Specificity and actionability that support downstream localization or patch generation.

For each version, provide a score from 1-10 and brief justification.
Then select the BEST version overall.

Format your response as:
### Original:
Score: X/10
Justification: [brief explanation]

### Variant 1:
Score: X/10
Justification: [brief explanation]

### Variant 2:
Score: X/10
Justification: [brief explanation]

### Variant 3:
Score: X/10
Justification: [brief explanation]

### Best Version: [Original/Variant 1/Variant 2/Variant 3]
Reason: [why this is the best]
"""
    
    response = run_model(prompt, max_output_tokens=3000)
    
    # Parse evaluation results
    score_pattern = r'### (Original|Variant \d+):\s*Score:\s*(\d+)/10\s*Justification:\s*(.*?)(?=###|$)'
    scores = re.findall(score_pattern, response, re.DOTALL)
    
    best_pattern = r'### Best Version:\s*(Original|Variant \d+)\s*Reason:\s*(.*?)$'
    best_match = re.search(best_pattern, response, re.DOTALL)
    
    evaluation = {
        'scores': {},
        'best_version': None,
        'best_reason': None,
        'raw_response': response
    }
    
    for version, score, justification in scores:
        evaluation['scores'][version] = {
            'score': int(score),
            'justification': justification.strip()
        }
    
    if best_match:
        evaluation['best_version'] = best_match.group(1).strip()
        evaluation['best_reason'] = best_match.group(2).strip()
    
    return evaluation


def refine_reasoning(
    reasoning_file_path: str,
    output_dir: str = "refined_reasoning",
    num_variants: int = 3
) -> Dict[str, Any]:
    """
    Main function: Read reasoning file and optimize step by step
    
    Args:
        reasoning_file_path: Path to reasoning JSON file
        output_dir: Output directory
        num_variants: Number of variants to generate for each step
    
    Returns:
        Optimization results
    """
    # Read original reasoning file
    with open(reasoning_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instance_id = data['instance_id']
    problem = data['problem']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'instance_id': instance_id,
        'problem': problem,
        'stages': {}
    }
    
    # Process reasoning for each stage (file_think, func_think, edit_think, patch_think)
    stages = ['file_think', 'func_think', 'edit_think', 'patch_think']
    ground_truth_mapping = {
        'file_think': f"Ground Truth Files: {data.get('gt_files', [])}",
        'func_think': f"Ground Truth Elements: {data.get('gt_related_elements', {})}",
        'edit_think': f"Ground Truth Edit Locations: {data.get('gt_edit_locs', '')}",
        'patch_think': f"Ground Truth Patch: {data.get('patch', '')}"
    }
    
    for stage in stages:
        if stage not in data or not data[stage]:
            print(f"⚠️  Stage '{stage}' not found or empty, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing stage: {stage}")
        print(f"{'='*60}")
        
        reasoning_text = data[stage]
        steps = parse_steps(reasoning_text)
        
        if not steps:
            print(f"⚠️  No steps found in {stage}, skipping...")
            continue
        
        print(f"Found {len(steps)} steps in {stage}")
        
        stage_results = {
            'original_reasoning': reasoning_text,
            'original_steps': steps,
            'refined_steps': []
        }
        
        # Prepare stage-specific context
        stage_context = {
            'repository_structure': data.get('repository_structure', ''),
            'file_skeleton': data.get('file_skeleton', ''),
            'file_content': data.get('file_content', '')
        }
        
        # Process each step
        for i, step_info in enumerate(steps):
            step_number = step_info['step_number']
            step_content = step_info['content']
            
            print(f"\n  Processing Step {step_number}...")
            
            # Get context
            context_before = '\n'.join([s['content'] for s in steps[:i]]) if i > 0 else ""
            context_after = '\n'.join([s['content'] for s in steps[i+1:]]) if i < len(steps)-1 else ""
            
            # Generate variants
            print(f"    Generating {num_variants} variants...")
            variants = generate_step_variants(
                step_content,
                step_number,
                context_before,
                context_after,
                problem,
                stage,
                stage_context,
                num_variants
            )
            
            print(f"    Generated {len(variants)} variants")
            
            # Compare and evaluate
            print(f"    Evaluating variants...")
            ground_truth = ground_truth_mapping.get(stage, "")
            evaluation = compare_step_variants(
                step_content,
                variants,
                step_number,
                problem,
                ground_truth
            )
            
            # Select best version
            best_version = evaluation.get('best_version', 'Original')
            if best_version == 'Original':
                best_content = step_content
            else:
                variant_num = int(best_version.split()[-1]) - 1
                best_content = variants[variant_num] if variant_num < len(variants) else step_content
            
            print(f"    ✓ Best version: {best_version}")
            
            refined_step = {
                'step_number': step_number,
                'original_content': step_content,
                'variants': variants,
                'evaluation': evaluation,
                'best_version': best_version,
                'best_content': best_content
            }
            
            stage_results['refined_steps'].append(refined_step)
        
        # Combine optimized reasoning
        refined_reasoning = '\n'.join([
            step['best_content'] for step in stage_results['refined_steps']
        ])
        stage_results['refined_reasoning'] = refined_reasoning
        
        results['stages'][stage] = stage_results
        
        print(f"\n✅ Stage '{stage}' completed!")
    
    # Save results
    output_file = os.path.join(output_dir, f"{instance_id}_refined.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ All stages completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return results


def batch_refine_reasoning(
    reasoning_dir: str = "gpt5/reasoning",
    output_dir: str = "gpt5/refined_reasoning",
    num_variants: int = 3
):
    """
    Batch process reasoning files
    
    Args:
        reasoning_dir: Directory of reasoning files
        output_dir: Output directory
        num_variants: Number of variants to generate for each step
    """
    if not os.path.exists(reasoning_dir):
        print(f"❌ Directory not found: {reasoning_dir}")
        return
    
    json_files = [f for f in os.listdir(reasoning_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ No JSON files found in {reasoning_dir}")
        return
    
    print(f"Found {len(json_files)} reasoning files to process")
    
    for json_file in json_files:
        reasoning_file_path = os.path.join(reasoning_dir, json_file)
        print(f"\n{'#'*80}")
        print(f"Processing: {json_file}")
        print(f"{'#'*80}")
        
        try:
            refine_reasoning(reasoning_file_path, output_dir, num_variants)
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'#'*80}")
    print(f"✅ Batch processing completed!")
    print(f"{'#'*80}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process single file
        reasoning_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "gpt5/refined_reasoning"
        refine_reasoning(reasoning_file, output_dir)
    else:
        # Batch processing
        print("Usage:")
        print("  Single file: python step_refine.py <reasoning_file.json> [output_dir]")
        print("  Batch mode: modify the script to call batch_refine_reasoning()")
        print("\nRunning in single file mode with example...")
        
        # Example: Process single file
        example_file = "gpt5/reasoning/django__django-14017.json"
        if os.path.exists(example_file):
            refine_reasoning(example_file, "gpt5/refined_reasoning")
        else:
            print(f"Example file not found: {example_file}")
            print("Please provide a reasoning file path as argument.")
