import requests
import time
from itertools import cycle

def get_diff_from_pr(repo, pr_number, tokens, max_retries=5):
    api_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    token_cycle = cycle(tokens)  
    
    retries = 0
    wait_time = 30  

    while retries < max_retries:
        current_token = next(token_cycle)
        headers = {
            "Authorization": f"token {current_token}",
            "User-Agent": "PRDiffBot",
            "Accept": "application/vnd.github.v3.diff"
        }

        print(f"[Using token] {current_token[:6]}...")

        try:
            response = requests.get(api_url, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.text

            elif response.status_code == 404:
                print("[404] Not Found.")
                return ""

            elif response.status_code in (403, 429):
                print(f"[{response.status_code}] API limit triggered. Attempting to switch Token...")
                remaining = response.headers.get('X-RateLimit-Remaining')
                reset = int(response.headers.get('X-RateLimit-Reset', 0))
                if remaining and reset:
                    to_wait = max(reset - int(time.time()) + 1, 5)
                    print(f"Rate limit reached. Will reset at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reset))}")
                    time.sleep(to_wait)
                else:
                    print(f"Unknown restriction type. Retry after {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time += 15  # Exponential backoff
                retries += 1

            else:
                print(f"[Error] Unexpected status code: {response.status_code}")
                retries += 1
                time.sleep(wait_time)
                wait_time += 15

        except requests.exceptions.RequestException as e:
            print(f"[Network Error] {e}")
            retries += 1
            time.sleep(wait_time)
            wait_time += 15

    print("[Error] Reaching the maximum retry count, unable to obtain the diff.")
    return ""



# --------------
def filter_test_nonpythonfile(file_path):
    if not file_path:
        return True  
    path_parts = file_path.split('/')
    if any(part.lower().startswith('test') for part in path_parts) or not file_path.endswith('.py'):
        # print("test file or non-python file, skip.",file_path)
        return True
    else:
        return False
def extract_file_from_patch(patch):
    files = []
    patch_lines = patch.split("\n")
    for line in patch_lines:
        if line.startswith("+++ b/") and filter_test_nonpythonfile(line[6:])!=True:
            files.append(line[6:])
    return files

def extract_function_from_patch(patch, exclude_func=filter_test_nonpythonfile):
    '''
    Extract the modified Python files and the defined functions or classes from the patch.
    Skip the test files and non-Python files. 
    Parameters:
        patch (str): Content of the Git patch
        exclude_func (function): Function that receives the file path and returns whether the file should be excluded 
    Return:
        func_loc_str: Formatting string
        file2func: Dictionary {File path: [Function/class name]}
    '''
    file2func = {}
    func_loc_str = ""
    current_file = None
    patch_lines = patch.split("\n")

    for line in patch_lines:
        if line.startswith("+++ b/"):
            file_path = line[6:]  
            if exclude_func(file_path):  
                current_file = None
            else:
                current_file = file_path
                file2func[current_file] = []
                func_loc_str += f"\n{current_file}\n"

        elif current_file is not None and line.startswith("@@"):
            code_line = line.split(" @@ ", 1)[-1]  

            if code_line.startswith("def "):
                func_name = code_line.split("def ", 1)[1].split("(", 1)[0].strip()
                func_entry = f"function: {func_name}"
                file2func[current_file].append(func_entry)
                func_loc_str += func_entry + "\n"

            elif code_line.startswith("class "):
                class_name = code_line.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip()
                class_entry = f"class: {class_name}"
                file2func[current_file].append(class_entry)
                func_loc_str += class_entry + "\n"

    func_loc_str = f"```\n{func_loc_str}\n```"
    return func_loc_str, file2func

def merge_line(line_loc_str):
    lines = line_loc_str.split("\n")
    new_lines = []
    for i in range(len(lines)):
        if i > 0 and lines[i].startswith("line: ") and lines[i-1].startswith("line: ") and int(lines[i].split(": ")[1]) == int(lines[i-1].split(": ")[1]) + 1:
            continue
        new_lines.append(lines[i])
    return "\n".join(new_lines)



def extract_line_from_patch(patch, exclude_func=filter_test_nonpythonfile):
    '''
    Extract the position of the first line of the modified source code from the patch.
    Skip test files and non-Python files. 
    Parameters:
        patch (str): Content of the Git patch
        exclude_func (function): Function that receives the file path and returns whether the file should be excluded 
    Return:
        str: Formatted string, including file path, function/class name and modified line number
    '''
    line_loc_str = ""
    current_file = None
    cur_line = 0
    func_str = ""
    in_addition_block = False  
    
    patch_lines = patch.split("\n")

    for line in patch_lines:
        if line.startswith("+++ b/"):
            file_path = line[6:]  
            if exclude_func(file_path):  
                current_file = None
            else:
                current_file = file_path
                line_loc_str += f"\n{current_file}\n"
                cur_line = 0
                func_str = ""
                in_addition_block = False

        elif current_file is not None:
            if line.startswith("@@"):
                # @@ -xx,yy +aa,bb @@ ...
                cur_line = int(line.split(" ")[2].split(",")[0][1:])  
                code_line = line.split(" @@ ", 1)[-1]  
                in_addition_block = False

                if code_line.startswith("def "):
                    func_str = "function: " + code_line.split("def ", 1)[1].split("(", 1)[0].strip()
                elif code_line.startswith("class "):
                    func_str = "class: " + code_line.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip()
                else:
                    func_str = ""

                if func_str:
                    line_loc_str += func_str + "\n"

            elif line.startswith("+ "):
                if not in_addition_block:
                    line_loc_str += f"line: {cur_line}\n"
                    in_addition_block = True
                cur_line += 1

            elif line.startswith("- "):
                in_addition_block = False

            elif line.startswith(" "):
                in_addition_block = False
                cur_line += 1

            elif not line.startswith("\\") and len(line) > 0:
                in_addition_block = False
                cur_line += 1

    line_loc_str = f"```\n{line_loc_str}\n```"
    line_loc_str = merge_line(line_loc_str)  
    return line_loc_str