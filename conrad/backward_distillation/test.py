import re

def parse_think_and_answer(response):

    if not response:
        return "", ""
    
    thinking_start = response.find("### Thinking:")
    answer_start = response.find("### Answer:")
    
    if thinking_start == -1 and answer_start == -1:
        return response.strip(), ""
    
    if thinking_start != -1 and answer_start != -1:
        thinking = response[thinking_start + len("### Thinking:"):answer_start].strip()
        answer = response[answer_start + len("### Answer:"):].strip()
    elif thinking_start != -1:
        thinking = response[thinking_start + len("### Thinking:"):].strip()
        answer = ""
    elif answer_start != -1:
        thinking = response[:answer_start].strip()
        answer = response[answer_start + len("### Answer:"):].strip()
    
    return thinking, answer

def _parse_model_return_lines(answer):

    if not answer or answer.strip() == "":
        return []
    
    import re
    
    answer = re.sub(r'```[a-zA-Z]*', '', answer)
    answer = re.sub(r'```', '', answer)
    answer = re.sub(r'`', '', answer)
    
    lines = answer.strip().split('\n')
    file_paths = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('#') or line.startswith('###'):
            continue
            
        line = line.lstrip('- ').lstrip('* ').lstrip('+ ').lstrip('> ').lstrip('• ')
        line = line.strip()
        
        if line and ('/' in line or line.endswith('.py')):
            line = line.strip('"\'.,')
            if line and '/' in line and '```' not in line:
                file_paths.append(line)
    
    seen = set()
    unique_paths = []
    for path in file_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    return unique_paths

if __name__ == "__main__":
    answer =  '''
    ```
src/flask/blueprints.py
fda
tests/test_blueprints.py

```
'''
    print(_parse_model_return_lines(answer))



    def build_reasoning_tree(self, problem: str, structure, gt_file:list, gt_related_elements:dict):
        root = SimpleTreeNode()

        valid_paths = collect_all_file_paths(structure) 

        structure_processed = show_project_structure(structure) 

        stage1_response_list = self.generate_reasoning_for_stage(
            problem, 1, repository_structure=structure_processed)   

        stage1_nodes = []
            
        for resp in stage1_response_list: 
            think, answer = parse_think_and_answer(resp)
            found_file_loc = _parse_model_return_lines(answer) 

            filtered_files = [f for f in found_file_loc if f in valid_paths]
            if not filtered_files:
                print(f"[Stage 1] No valid files found in {found_file_loc}, adding empty list instead.")
                filtered_files = []  

                
            print(f"[Stage 1] Thinking parsing: {think}","\n",f"[Stage 1] Answer parsing: {answer}.")
            print(f"[Stage 1] Found file loc: {found_file_loc}")
            temp_node = SimpleTreeNode(found_file_loc=filtered_files, stage=1)
            if not matches_ground_truth(temp_node, gt_file, gt_related_elements):
                print(f" Stage 1 response skipped: does not match ground truth.")
                continue  
            node = root.add_child(
                reasoning=think,
                answer = answer,
                trajectory=resp,
                found_file_loc=found_file_loc
            )
            node.print_detailed()
            stage1_nodes.append(node)


        stage2_nodes = []
        for n1 in stage1_nodes:
            ctx1 = n1.get_reasoning_context()
            previous_result = n1.answer
            predicted_files = n1.found_file_loc 
            file_contents = get_repo_files(structure, predicted_files)
            skeleton = localize_function_from_compressed_files(structure, predicted_files) #获取skeleton的结构

            stage2_response_list = self.generate_reasoning_for_stage(
                problem, 2, ctx1, previous_results=previous_result, file_skeleton=skeleton)

            for resp in stage2_response_list:
                print("stage 2 response:",resp)
                think, answer = parse_think_and_answer(resp) 
                found_related_loc = extract_code_blocks(resp) 
                print("found related locs:",found_related_loc)
                found_related_locs_separated = extract_locs_for_files(found_related_loc,predicted_files,False) #从找到的代码块中提取出文件名和function/class name.
                print("found related locs separated:",found_related_locs_separated)
                print(f"Stage 2 - Raw answer: {answer}")
                print(f"Stage 2 - Predicted files: {predicted_files}")
                print(f"Debug - found_related_loc: {found_related_loc}")
                print(f"Debug - found_related_locs_separated: {found_related_locs_separated}")
                temp_node = SimpleTreeNode(
                    stage=2,
                    found_file_loc=n1.found_file_loc,
                    found_related_loc=found_related_locs_separated
                )
                if not matches_ground_truth(temp_node, gt_file, gt_related_elements):
                    print(f" Stage 2 response skipped: does not match ground truth.")
                    continue
                node = n1.add_child(
                    reasoning=think,
                    answer = answer,
                    trajectory = resp,
                    found_file_loc = n1.found_file_loc,
                    found_related_loc=found_related_locs_separated
)
                node.print_detailed()
                stage2_nodes.append(node)

        stage3_nodes = []
        for n2 in stage2_nodes:
            ctx2 = n2.get_reasoning_context() 
            previous_result = n2.answer
            pred_files_locs = n2.found_file_loc 
            found_related_locs = n2.found_related_loc 
            file_contents = get_repo_files(structure, predicted_files)
            # predicted_locs = extract_locations_from_reasoning(n2.reasoning)
            topn_content, file_loc_intervals = construct_topn_file_context( 
                found_related_locs,
                pred_files_locs,
                file_contents,
                structure,
                context_window=10,
                loc_interval=True,
                no_line_number=False,
            )

            stage3_response_list = self.generate_reasoning_for_stage( 
                problem, 3, ctx2, previous_results=previous_result, file_contents=topn_content)

            for resp in stage3_response_list: 
                think, answer = parse_think_and_answer(resp) 
                found_edit_locs = extract_code_blocks(answer)
                found_edit_locss_separated = extract_locs_for_files( 
                    found_edit_locs, pred_files_locs, False
                )
                print("raw_response",resp,"\n","think",think,"\n","answer",answer,"\n","found_edit_locs",found_edit_locs,"\n","found_edit_locss_separated",found_edit_locss_separated,"\n")
                # found_edit_locs = parse_stage3_answer(answer)
                node = n2.add_child(
                    reasoning=think,
                    answer = answer,
                    trajectory = resp,
                    found_file_loc = n2.found_file_loc,
                    found_related_loc=n2.found_related_loc,
                    found_edit_locs=found_edit_locss_separated
                )
                node.print_detailed()
                stage3_nodes.append(node)

        stage4_nodes = []
        for n3 in stage3_nodes:
            ctx3 = n3.get_reasoning_context()
            edit_targets = n3.found_edit_locs
            # code_content = get_code_content(edit_targets)
            code_content = '...'
            stage4_response_list = self.generate_reasoning_for_stage(
                problem, 4, ctx3, code_content=code_content)

            for resp in stage4_response_list:
                think, answer = parse_think_and_answer(resp)
                node = n3.add_child(
                    reasoning=think,
                    answer = answer,
                    trajectory = resp,
                    found_file_loc = n3.found_file_loc,
                    found_related_loc=n3.found_related_loc,
                    found_edit_locs=n3.found_edit_locs
                )
                node.print_detailed()
                stage4_nodes.append(node)

        return root, stage4_nodes



def parse_think_and_answer(response):

    if not response:
        return "", ""
    
    thinking_start = response.find("### Thinking:")
    answer_start = response.find("### Answer:")
    
    if thinking_start == -1 and answer_start == -1:
        return response.strip(), ""
    
    if thinking_start != -1 and answer_start != -1:
        thinking = response[thinking_start + len("### Thinking:"):answer_start].strip()
        answer = response[answer_start + len("### Answer:"):].strip()
    elif thinking_start != -1:
        thinking = response[thinking_start + len("### Thinking:"):].strip()
        answer = ""
    elif answer_start != -1:
        thinking = response[:answer_start].strip()
        answer = response[answer_start + len("### Answer:"):].strip()
    
    return thinking, answer

def collect_all_file_paths(structure, prefix=""):
    """Recursively collect all valid file paths from the nested structure dict."""
    file_paths = set()
    for name, content in structure.items():
        if isinstance(content, dict):
            if all(k in content for k in ("classes", "functions", "text")):
                file_paths.add(f"{prefix}{name}")
            else:
                file_paths.update(collect_all_file_paths(content, prefix=f"{prefix}{name}/"))
    return file_paths

def _parse_model_return_lines(answer):

    if not answer or answer.strip() == "":
        return []
    
    import re
    
    answer = re.sub(r'```[a-zA-Z]*', '', answer)
    answer = re.sub(r'```', '', answer)
    answer = re.sub(r'`', '', answer)
    
    lines = answer.strip().split('\n')
    file_paths = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('#') or line.startswith('###'):
            continue
            

        line = line.lstrip('- ').lstrip('* ').lstrip('+ ').lstrip('> ').lstrip('• ')
        line = line.strip()
        

        if line and ('/' in line or line.endswith('.py')):
            line = line.strip('"\'.,')
            if line and '/' in line and '```' not in line:
                file_paths.append(line)
    
    
    seen = set()
    unique_paths = []
    for path in file_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    return unique_paths

def extract_code_blocks(text):
    pattern = r"```\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        if "```" in text:
            # handle the case where the code block is not complete
            return [text.split("```", 1)[-1].strip()]
    return matches
def extract_locs_for_files(locs, file_names, keep_old_order=False):
    if keep_old_order:
        results = {fn: [] for fn in file_names}
    else:
        results = {}  # dict is insertion ordered
    current_file_name = None
    for loc in locs:
        for line in loc.splitlines():
            if line.strip().endswith(".py"):
                current_file_name = line.strip()
            elif line.strip() and any(
                line.startswith(w)
                for w in ["line:", "function:", "class:", "variable:"]
            ):
                if current_file_name in file_names:
                    if current_file_name not in results:
                        results[current_file_name] = []
                    results[current_file_name].append(line)
                else:
                    pass
    for file_name in file_names:
        if file_name not in results:  # guard for new order case
            results[file_name] = []
    return {fn: ["\n".join(results[fn])] for fn in results.keys()}

