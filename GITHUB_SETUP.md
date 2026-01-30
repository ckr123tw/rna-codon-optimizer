# Complete Git Setup and GitHub Upload Guide

## ⚠️ STEP 0: Install Git (REQUIRED FIRST!)

Git is not installed on your system. You need to install it first.

### Install Git for Windows

**Option 1: Using winget (Recommended)**
```powershell
winget install --id Git.Git -e --source winget
```

**Option 2: Direct Download**
1. Download from: https://git-scm.com/download/win
2. Run the installer
3. Use default settings (just click "Next")

**After installation:**
- **CLOSE and REOPEN PowerShell** (important!)
- Verify: `git --version`

---

## STEP 1: Configure Git (First Time Only)

```powershell
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Configure credential storage
git config --global credential.helper manager-core
```

---

## STEP 2: Initialize Git Repository

```powershell
# Navigate to project
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer

# Initialize Git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: RNA codon optimization pipeline with multi-metric support"
```

---

## STEP 3: Create GitHub Repository

1. **Go to:** https://github.com/new
2. **Repository name:** `rna-codon-optimizer`
3. **Description:** "AI-powered RNA sequence optimization using reinforcement learning"
4. **Visibility:** Public (or Private)
5. **License:** None (we have our own)
6. **DO NOT** check "Add README" or "Add .gitignore"
7. Click **"Create repository"**

---

## STEP 4: Connect to GitHub

After creating the repo on GitHub, you'll see setup instructions. Use these commands:

```powershell
# Add remote
git remote add origin https://github.com/ckr123tw/rna-codon-optimizer.git

# Rename branch to main
git branch -M main
```

---

## STEP 5: Create Personal Access Token

**GitHub doesn't accept passwords. You need a token.**

1. **Create token:** https://github.com/settings/tokens/new
2. **Settings:**
   - Note: `rna-codon-optimizer`
   - Expiration: 90 days
   - ✅ Check `repo` (Full control)
3. Click **"Generate token"**
4. **COPY THE TOKEN** (starts with `ghp_`)
5. **Save it somewhere safe** - you won't see it again!

---

## STEP 6: Push to GitHub

```powershell
# Push to GitHub
git push -u origin main
```

**When prompted:**
- **Username:** `ckr123tw`
- **Password:** `ghp_xxxxxxx...` (PASTE YOUR TOKEN, NOT PASSWORD!)

---

## STEP 7: Verify

Visit: https://github.com/ckr123tw/rna-codon-optimizer

You should see all your files!

---

## Alternative: Use GitHub Desktop (No Command Line)

If you prefer a GUI:

1. **Download:** https://desktop.github.com/
2. **Install and sign in**
3. **File → Add Local Repository**
4. Select: `C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer`
5. **Publish repository**

---

## Quick Troubleshooting

### "git: command not found"
- Install Git (Step 0)
- Restart PowerShell

### "not a git repository"
- Run `git init` in the project folder
- Make sure you're in the right directory: `pwd`

### "Password authentication not supported"
- Use Personal Access Token instead of password
- Follow Step 5

### "fatal: repository not found"
- Create the repository on GitHub first (Step 3)
- Check the URL is correct

### "Permission denied"
- Make sure you have access to the repository
- Check your token has `repo` scope

---

## Complete Example (All Steps)

```powershell
# 0. Install Git first (winget or download)
winget install --id Git.Git -e --source winget

# Close and reopen PowerShell!

# 1. Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global credential.helper manager-core

# 2. Initialize repository
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
git init
git add .
git commit -m "Initial commit: RNA codon optimization pipeline"

# 3. Create repo on GitHub (via web interface)
# Visit: https://github.com/new

# 4. Connect and push
git remote add origin https://github.com/ckr123tw/rna-codon-optimizer.git
git branch -M main
git push -u origin main

# When prompted, use your username and PERSONAL ACCESS TOKEN
```

---

## Need Help?

If you get stuck, you can use GitHub CLI instead:

```powershell
# Install GitHub CLI
winget install --id GitHub.cli

# Authenticate (opens browser)
gh auth login

# Create and push repo in one command
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
gh repo create rna-codon-optimizer --public --source=. --remote=origin --push
```

This is the easiest method if you're having authentication issues!
