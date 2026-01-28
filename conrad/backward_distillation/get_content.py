import requests
from unidiff import PatchSet
from collections import defaultdict
import json
from util.compress_file import get_skeleton
from util.postprocess_data import extract_code_blocks, extract_locs_for_files
from util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)
def extract_patches(repo,instance_id) -> tuple[str, str]:
    """
    Get patch and test patch from PR

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """
    patch = requests.get(f"https://github.com/{repo}/pull/{instance_id}.diff").text
    patch_test = ""
    patch_fix = ""
    for hunk in PatchSet(patch):
        if any(
            test_word in hunk.path for test_word in ["test", "tests", "e2e", "testing"]
        ):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix

def extract_function_from_patch(patch): 
    file2func = {}
    func_loc_str = ""
    file = ""
    func_str = ""
    patch_lines = patch.split("\n")
    for line in patch_lines:
        if line.startswith("+++ b/"):
            # files.append(line[6:])
            func_loc_str += "\n" + line[6:] + "\n"
            file2func[line[6:]] = []
            file = line[6:]
        if line.startswith("@@"):
            line = line.split(" @@ ")[-1]
            if line.startswith("def "):
                func_str = "function: " + line.split("def ")[-1].split("(")[0]
                file2func[file].append(func_str)
            if line.startswith("class "):
                func_str = "class: " + line.split("class ")[-1].split("(")[0].split(":")[0]
                file2func[file].append(func_str)
            if func_str:
                func_loc_str += func_str + "\n"
                func_str = ""
    func_loc_str = f"```\n{func_loc_str}\n```"
    return func_loc_str, file2func




#从structure的某个文件里获取所有内容.
def get_full_file_paths_and_classes_and_functions(structure, current_path=""):
    """
    Recursively retrieve all file paths, classes, and functions within a directory structure.

    Arguments:
    structure -- a dictionary representing the directory structure
    current_path -- the path accumulated so far, used during recursion (default="")

    Returns:
    A tuple containing:
    - files: list of full file paths
    - classes: list of class details with file paths
    - functions: list of function details with file paths
    """
    files = []
    classes = []
    functions = []
    for name, content in structure.items():
        if isinstance(content, dict):
            if (
                not "functions" in content.keys()
                and not "classes" in content.keys()
                and not "text" in content.keys()
            ) or not len(content.keys()) == 3:
                # or guards against case where functions and classes are somehow part of the structure.
                next_path = f"{current_path}/{name}" if current_path else name
                (
                    sub_files,
                    sub_classes,
                    sub_functions,
                ) = get_full_file_paths_and_classes_and_functions(content, next_path)
                files.extend(sub_files)
                classes.extend(sub_classes)
                functions.extend(sub_functions)
            else:
                next_path = f"{current_path}/{name}" if current_path else name
                files.append((next_path, content["text"]))
                if "classes" in content:
                    for clazz in content["classes"]:
                        classes.append(
                            {
                                "file": next_path,
                                "name": clazz["name"],
                                "start_line": clazz["start_line"],
                                "end_line": clazz["end_line"],
                                "methods": [
                                    {
                                        "name": method["name"],
                                        "start_line": method["start_line"],
                                        "end_line": method["end_line"],
                                    }
                                    for method in clazz.get("methods", [])
                                ],
                            }
                        )
                if "functions" in content:
                    for function in content["functions"]:
                        function["file"] = next_path
                        functions.append(function)
        else:
            next_path = f"{current_path}/{name}" if current_path else name
            files.append(next_path)
    return files, classes, functions

def get_repo_files(structure, filepaths: list[str]):
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for filepath in filepaths:
        content = None

        for file_content in files:
            if file_content[0] == filepath:
                content = "\n".join(file_content[1])
                file_contents[filepath] = content
                break
        # print('filepath:',filepath)
        if content is None:
            print(f"Warning: file not found, skip {filepath}")
            continue  # 跳过该文件，不报错
    return file_contents


#skeleton:

file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""
def localize_function_from_compressed_files(
    structure,
    file_names,
    mock=False,
    temperature=0.0,
    keep_old_order=False,
    compress_assign: bool = False,
    total_lines=30,
    prefix_lines=10,
    suffix_lines=10,
    ):
    file_contents = get_repo_files(structure, file_names)
    compressed_file_contents = {
        fn: get_skeleton(
            code,
            compress_assign=compress_assign,
            total_lines=total_lines,
            prefix_lines=prefix_lines,
            suffix_lines=suffix_lines,
        )
        for fn, code in file_contents.items()
    }
    contents = [
        file_content_in_block_template.format(file_name=fn, file_content=code)
        for fn, code in compressed_file_contents.items()
    ]
    file_contents = "".join(contents)
    return file_contents

from util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    line_wrap_content,
    transfer_arb_locs_to_locs,
)

def construct_topn_file_context(
    file_to_locs, #coarse_found_locs = found_related_locs
    pred_files, #pred_files = file_loc
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.
    
    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    print(f"file_to_locs: {file_to_locs}")
    print(f"pred_files: {pred_files}")
    print(f"file_contents keys: {file_contents.keys()}")
    print(f"structure type: {type(structure)}")
    file_loc_intervals = dict()
    topn_content = ""
    for pred_file, locs in file_to_locs.items():
        if pred_file not in file_contents:
            print(f"Warning: {pred_file} is not found in file_contents.")
            continue

        content = file_contents[pred_file]
        print(f"construct function__content: {content}")
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )
        print(f"line_locs: {line_locs}")
        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals
