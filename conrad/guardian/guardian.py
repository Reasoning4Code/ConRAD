from __future__ import annotations
import json, re
from typing import List, Dict, Any, Callable

# ------------------------------
# Data Structure
# ------------------------------
# candidates: List[Dict], each element contains at least:
#   {"id": "c1", "issue_summary": "...", "patch": "..."}   # patch can be empty

# ------------------------------
# Prompt Generation
# ------------------------------
def build_batch_prompt(current_issue_summary: str,
                       current_patch: str,
                       candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Returns messages (system+user) required for Chat Completions.
    candidates list can be 1~N (N<=5), if empty will require returning {"candidates": []}
    """
    # Combine candidate text blocks
    cand_blocks = []
    for idx, c in enumerate(candidates, start=1):
        cid = c.get("id", f"{idx}")
        csum = c.get("summary", "").strip()
        cpatch = (c.get("patch") or "").strip()
        
        # If patch is empty, try to extract Changes Summary from prompt_used
        if not cpatch:
            prompt_used = c.get("prompt_used", "")
            if "Changes Summary:" in prompt_used:
                # Extract Changes Summary section
                start = prompt_used.find("Changes Summary:")
                if start != -1:
                    # Find the end of Changes Summary (usually end of string)
                    changes_section = prompt_used[start:].strip()
                    # Remove "Changes Summary:" title, keep only content
                    cpatch = changes_section.replace("Changes Summary:", "").strip()
        
        # Get original issue title and body
        similar_issue = c.get("similar_issue", {})
        issue_title = similar_issue.get("issue_title", "").strip()
        issue_body = similar_issue.get("issue_body", "").strip()
        
        # Build complete description
        description = f"Title: {issue_title}\n\nBody: {issue_body}" if issue_title or issue_body else "No description available"
        
        block = (
            f"[{idx}] <Candidate id={cid}>\n"
            f"Original Description:\n{description}\n\n"
            f"Issue Summary:\n{csum}\n\n"
            f"Code Changes (for understanding only):\n{cpatch}\n"
            f"</Candidate id={cid}>\n"
        )
        cand_blocks.append(block)

    user_text = f"""
You are an expert software debugging analyst specializing in identifying transferable bug-fixing patterns.

TASK:
Evaluate this historical issue to determine whether its debugging approach, reasoning pattern, or fix strategy can guide solving the CURRENT issue.
We want to include the historical issue as a few-shot example to guide on fixing the current issue. Classify each as **Useful**, **Not useful**, or **Harmful** based on transferability and potential to mislead. 

IMPORTANT CALIBRATION:
Historical candidates are retrieved automatically and may be noisy.
Do NOT default to "Useful".
Default to "Not useful" unless you can point to at least ONE concrete, candidate-specific transferable aspect
that clearly connects to the CURRENT issue (root cause / mechanism / component / patch-shape / test pattern).

Weak usefulness is allowed, but ONLY when there is concrete evidence from the historical issue.
Generic debugging advice (e.g., "write a regression test", "trace internal path") WITHOUT candidate-specific evidence
is NOT sufficient for "Useful".

<CURRENT_ISSUE>
Issue Summary:
{current_issue_summary.strip()}
</CURRENT_ISSUE>

<HISTORICAL_ISSUES>
{''.join(cand_blocks) if cand_blocks else '(no candidates provided)'}
</HISTORICAL_ISSUES>

------------------------------------------------------------
EVALUATION CRITERIA  (in priority order)
------------------------------------------------------------

1. **ROOT CAUSE SIMILARITY**  *(Highest Priority)*
   - Does the historical issue share the same fundamental problem type?
   - Examples: null pointer, race condition, off-by-one error, resource leak, logic inversion, API misuse, state management issue.

2. **CAUSAL CHAIN TRANSFERABILITY**
   - Does the historical issue demonstrate a similar sequence of events leading to failure?
   - Can the diagnostic reasoning path be reused effectively?
   - If both the historical and current issues involve a system's automated or implicit behavior suppressing or overriding explicit developer intent, treat the pair as potentially transferable, even if they occur in different subsystems or technologies.
        This includes cases where:
        Automatically generated logic, caching, schema evolution, or reflection-based mechanisms ignore, replace, or bypass explicit user configuration or overrides.
        A newer system version introduces hidden precedence or dispatch changes that silently alter user-visible behavior.
        The debugging process required inspecting generated code paths, internal dispatch layers, or meta-level logic to restore expected control flow.

3. **FIX STRATEGY APPLICABILITY**
   - Is the general solution approach (not necessarily the code) relevant?
   - Examples: adding validation, reordering operations, caching, defensive copying, changing data structures.

4. **CONTEXTUAL ALIGNMENT**
   - Are the system components, architectural layers, or runtime contexts aligned?
   - Examples: UI layer, database access, API handling, concurrency, initialization logic.

5. **DEBUGGING TECHNIQUE VALUE**
   - Does the historical issue reveal useful diagnostic or investigation methods?
   - Examples: specific test cases, reproduction steps, or diagnostic heuristics.

------------------------------------------------------------
DECISION GUIDELINES
------------------------------------------------------------

### Mark **Useful** ONLY if:
- You can extract ≥1 concrete, candidate-specific transferable aspect grounded in the historical issue AND
  it matches the current issue in at least ONE of the following:
  (1) root cause similarity, OR
  (2) explicit shared mechanism pattern (e.g., descriptor/property/metaclass dispatch, caching/registry precedence) mentioned in BOTH,
  (3) component/layer overlap (same module family / same runtime layer / same API boundary),
  (4) a clearly reusable patch-shape that applies to the same type of bug (e.g., precedence ordering, guard condition) with evidence.

Workflow-only transfer (repro → locate → minimal fix) WITHOUT any of (1)-(4) is NOT sufficient for Useful.
In that case, label as "Not useful" with usefulness_score in (0.0, 0.1].

### Mark **Harmful** if:
- The candidate would likely push toward a SPECIFIC wrong fix direction/layer/module (e.g., schema editor changes for a runtime dispatch bug),
  OR suggests a misleading patch category that wastes time.
(Mere subsystem difference is NOT harmful.)

### Mark **Not useful** if:
- You cannot name ANY actionable transferable aspect (where-to-look / what-to-trace / what-to-test / patch-shape).
- No shared mechanism pattern can be articulated.
- The issue is too vague or unrelated such that it offers neither workflow nor diagnostic reuse.

------------------------------------------------------------
CLASSIFICATION PRIORITY & SCORING
------------------------------------------------------------

- If both *Useful* and *Harmful* aspects exist → **Mark "Harmful"** (safety first).
- If uncertain between *Harmful* and *Not useful*, consider whether the mismatch could mislead.
  - If yes → **Harmful**.
  - If truly neutral → **Not useful**.


**Usefulness Score Range:**
- Harmful: **[-1.0, -0.1]**
  - ≤ −0.5 when the historical bug belongs to a completely different subsystem and could mislead debugging.
- Not useful: **(−0.1, 0.1)**
- Useful: **[0.1, 1.0]**

------------------------------------------------------------
OUTPUT FORMAT  *(strict JSON only)*
------------------------------------------------------------

{{
  "candidates": [
    {{
      "idx": <int>,
      "id": "<original_id>",
      "decision": "Useful" | "Not useful" | "Harmful",
      "usefulness_score": <float between -1.0 and 1.0>,
      "confidence_score": <float between 0.0 and 1.0>,
      "transferable_aspects": [<list of 1–3 concrete items: for Useful → what transfers; for Harmful → what misleads; for Not useful → can be empty array>],
      "reason": "<1–2 concise sentences explaining WHY, referencing the above criteria. For Harmful, explicitly state what could mislead.>"
    }}
  ]
}}

------------------------------------------------------------
CRITICAL RULES
------------------------------------------------------------
- EVIDENCE RULE: Every item in "transferable_aspects" must be grounded in the historical issue content
  (e.g., mentions a specific mechanism, module/layer, failure mode, test pattern, or patch shape).
  If you cannot ground it, do NOT include it and do NOT label the candidate as Useful based on it.
- Before choosing "Not useful", attempt to find a candidate-specific transferable aspect.
  If you cannot ground any aspect in the historical issue, explicitly state "no grounded transferable aspect".
- Evaluate **every** candidate [1..N]; return empty array if none.
- Output **valid JSON only** — no Markdown or extra text.
- Be **specific** in reasoning; vague or abstract explanations reduce utility.
- Emphasize **root cause + context alignment** over superficial symptom overlap.
- Be liberal in identifying potentially **Useful** reasoning but conservative about **Harmful** ones.
""".strip()





    system_text = (
        "You are an impartial software reasoning evaluator. "
        # "Answer deterministically and be concise. If unsure, choose 'Not useful' with Low usefulness_score."
        "Answer deterministically and be concise."
    )

    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    }

# ------------------------------
# Self Reflection Prompt Generation
# ------------------------------
def build_reflection_prompt(original_prompt: str, initial_judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build simple self reflection prompt to let LLM re-examine its judgment
    """
    
    # Format initial judgment result as JSON string
    initial_json = json.dumps({
        "candidates": initial_judgment.get("candidates", [])
    }, indent=2, ensure_ascii=False)
    
    reflection_text = f"""
{original_prompt}

Your previous answer was:
{initial_json}

Can you review your decision critically and give an updated answer? Please use the exact same JSON format as before.

{{
  "candidates": [
    {{
      "idx": <int>,
      "id": "<original_id>",
      "decision": "Useful" | "Not useful" | "Harmful",
      "usefulness_score": <float between -1.0 and 1.0>,
      "confidence_score": <float between 0.0 and 1.0>,
      "transferable_aspects": [<list of 1–3 concrete items>],
      "reason": "<your reasoning>"
    }}
  ]
}}

Output ONLY valid JSON, no additional text.
""".strip()
    
    system_text = (
        "You are performing critical self-reflection on your previous judgment. "
        "Be honest about potential errors and focus on improving accuracy."
    )
    
    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": reflection_text},
        ]
    }

# ------------------------------
# Output Parsing (Robust JSON Extraction)
# ------------------------------
def parse_json_strict(s: str) -> Dict[str, Any]:
    # Direct attempt
    try:
        return json.loads(s)
    except Exception:
        # Try to extract outermost JSON block from text
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise ValueError("No JSON found in model output")
        return json.loads(m.group(0))

# ------------------------------
# Unified Entry Point (LLM call injected externally)
# ------------------------------
def judge_batch_with_reflection(
    llm: Callable[[Dict[str, Any]], str],
    current_issue_summary: str,
    current_patch: str,
    candidates: List[Dict[str, str]],
    use_reflection: bool = True
) -> Dict[str, Any]:
    """
    Judgment process with self reflection
    
    Args:
        llm: LLM call function
        current_issue_summary: Current issue summary
        current_patch: Current issue patch
        candidates: List of candidate historical issues
        use_reflection: Whether to use self reflection
    
    Returns:
        dict: Complete result containing initial judgment, reflection result, and final judgment
    """
    
    # Phase 1: Initial judgment
    initial_result = judge_batch(llm, current_issue_summary, current_patch, candidates)
    
    result = {
        "initial_judgment": initial_result,
        "reflection_used": use_reflection
    }
    
    if not use_reflection:
        result["final_judgment"] = initial_result
        return result
    
    # Phase 2: Self Reflection
    try:
        # Build reflection prompt
        original_prompt_data = build_batch_prompt(current_issue_summary, current_patch, candidates)
        original_prompt = original_prompt_data["messages"][1]["content"]
        
        reflection_prompt = build_reflection_prompt(original_prompt, initial_result)
        reflection_raw = llm(reflection_prompt)
        reflection_result = parse_json_strict(reflection_raw)
        
        # Process reflection result - use same format as initial judgment
        final_judgment = {
            "candidates": []
        }
        
        # Process reflection results for each candidate
        by_idx = {i+1: c for i, c in enumerate(candidates)}
        for item in reflection_result.get("candidates", []):
            idx = int(item.get("idx", -1))
            if idx in by_idx:
                final_judgment["candidates"].append({
                    "idx": idx,
                    "id": item.get("id", by_idx[idx].get("id","")),
                    "decision": item.get("decision", ""),
                    "usefulness_score": item.get("usefulness_score", ""),
                    "confidence_score": item.get("confidence_score", ""),
                    "transferable_aspects": item.get("transferable_aspects", []),
                    "reason": item.get("reason", "").strip()
                })
        
        # Fill in missing candidates
        expected = set(range(1, len(candidates)+1))
        covered = {x["idx"] for x in final_judgment["candidates"]}
        for idx in sorted(expected - covered):
            # Use initial judgment as default
            initial_candidate = next((c for c in initial_result["candidates"] if c["idx"] == idx), None)
            if initial_candidate:
                final_judgment["candidates"].append(initial_candidate)
        
        final_judgment["candidates"].sort(key=lambda x: x["idx"])
        result["reflection_judgment"] = final_judgment
        result["final_judgment"] = result["reflection_judgment"]
        
    except Exception as e:
        print(f"  Self reflection failed, using initial judgment: {e}")
        result["reflection_error"] = str(e)
        result["final_judgment"] = initial_result
    
    return result

def judge_batch(
    llm: Callable[[Dict[str, Any]], str],
    current_issue_summary: str,
    current_patch: str,
    candidates: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    llm: Receives {'messages': [...]}, returns string (model's content).
    Returns dict: {"candidates":[{idx,id,decision,confidence,reason}, ...]}
    """
    payload = build_batch_prompt(current_issue_summary, current_patch, candidates)
    raw = llm(payload)  # Integrate with OpenAI/DeepSeek/Claude etc.
    data = parse_json_strict(raw)

    # Fallback cleanup: keep only necessary fields and ensure order matches input
    out = {"candidates": []}
    by_idx = {i+1: c for i, c in enumerate(candidates)}
    for item in data.get("candidates", []):
        idx = int(item.get("idx", -1))
        if idx in by_idx:
            out["candidates"].append({
                "idx": idx,
                "id": item.get("id", by_idx[idx].get("id","")),
                "decision": item.get("decision", ""),
                "usefulness_score": item.get("usefulness_score", ""),
                "confidence_score": item.get("confidence_score", ""),  # New field
                "transferable_aspects": item.get("transferable_aspects", []),
                "reason": item.get("reason", "").strip(),
            })
    # If model misses some candidates, fill them with default "Not useful"
    expected = set(range(1, len(candidates)+1))
    covered = {x["idx"] for x in out["candidates"]}
    for idx in sorted(expected - covered):
        out["candidates"].append({
            "idx": idx,
            "id": by_idx[idx].get("id",""),
            "decision": "Not useful",
            "usefulness_score": 0.0,
            "confidence_score": 0.5,  # Default medium confidence
            "transferable_aspects": [],
            "reason": "No judgment returned; defaulted to conservative decision."
        })
    # If no candidates this round
    if not candidates:
        out = {"candidates": []}
    # Sort by idx
    out["candidates"].sort(key=lambda x: x["idx"])
    return out

# ------------------------------
# Example: Integration with OpenAI (can be replaced with any LLM)
# ------------------------------
# pip install openai>=1.0.0
# from openai import OpenAI
# client = OpenAI(api_key="sk-...")

# def openai_llm(payload: Dict[str, Any]) -> str:
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=payload["messages"],
#         temperature=0,
#     )
#     return resp.choices[0].message.content

# Usage example:
# result = judge_batch(openai_llm, current_issue_summary, current_patch, candidates)
# print(result)