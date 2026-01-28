# API Configuration Guide

This guide explains how to configure API keys for the ConRAD project.

## Quick Setup

### Step 1: Create `.env` file

Copy the example file and edit it with your API keys:

```bash
cd /Users/macbook/repo
cp .env.example .env
```

### Step 2: Edit `.env` file

Open `.env` in your text editor and add your API keys:

```bash
# Required for all scripts
OPENAI_API_KEY=sk-your-actual-openai-key

# Optional - only if using DeepSeek in guardian
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key

# Required for backward_distillation.py
GITHUB_TOKEN=ghp_your-actual-github-token
```

### Step 3: Install dependencies

Make sure you have all required packages:

```bash
pip install -r requirements.txt
```

## Where to Get API Keys

### OpenAI API Key
1. Visit: https://platform.openai.com/api-keys
2. Sign in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-...` or `sk-...`)
5. Paste it in `.env` file as `OPENAI_API_KEY=sk-...`

### DeepSeek API Key (Optional)
1. Visit: https://platform.deepseek.com/api_keys
2. Sign in or create account
3. Create a new API key
4. Copy and paste in `.env` file as `DEEPSEEK_API_KEY=sk-...`

### GitHub Personal Access Token
1. Visit: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name (e.g., "ConRAD Research")
4. Select scope: **`repo`** (Full control of private repositories)
5. Click "Generate token"
6. Copy the token (starts with `ghp_...`)
7. Paste in `.env` file as `GITHUB_TOKEN=ghp_...`

## Security Notes

⚠️ **IMPORTANT**: 
- **NEVER** commit your `.env` file to git
- The `.env` file is already in `.gitignore` to prevent accidental commits
- Keep your API keys secret and never share them
- If you accidentally expose a key, revoke it immediately and generate a new one

## Which Script Uses Which Key

| Script | Required Keys |
|--------|---------------|
| `exemplar_mining/llm_judge.py` | `OPENAI_API_KEY` |
| `guardian/run_guardian.py` | `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` |
| `backward_distillation/backward_distillation.py` | `OPENAI_API_KEY` + `GITHUB_TOKEN` |

## Testing Your Configuration

After setting up `.env`, test if it works:

```bash
# Test that environment variables are loaded
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

If you see "OpenAI Key: SET", your configuration is correct!

## Troubleshooting

### Error: "OPENAI_API_KEY environment variable not set"
- Make sure you created `.env` file in the project root directory
- Check that the key name is exactly `OPENAI_API_KEY` (case-sensitive)
- Ensure there are no spaces around the `=` sign

### Error: "Import 'dotenv' could not be resolved"
```bash
pip install python-dotenv
```

### Error: "GITHUB_TOKEN not set"
- Only needed for `backward_distillation.py`
- Follow the GitHub token setup instructions above
