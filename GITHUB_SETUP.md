# GitHub Setup Guide

## Prerequisites

1. **Install Git**
   - Download from: https://git-scm.com/download/win
   - Or use: `winget install --id Git.Git -e --source winget`
   - Restart your terminal after installation

2. **GitHub Account**
   - Create account at: https://github.com
   - Configure Git:
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "your.email@example.com"
     ```

## Upload to GitHub

### Option 1: Create New Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `rna-codon-optimizer`
3. Description: "AI-powered RNA sequence optimization using reinforcement learning"
4. Keep it Public (or Private)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Option 2: Push to Existing Repository

If you already have a repository at https://github.com/ckr123tw/rna-codon-optimizer:

```bash
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: RNA codon optimization pipeline with multi-metric support"

# Add remote (replace with your actual repo URL)
git remote add origin https://github.com/ckr123tw/rna-codon-optimizer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 3: Create Completely New Repository

```bash
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer

# Initialize
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: RNA codon optimization pipeline"

# Create new repo on GitHub first, then:
git remote add origin https://github.com/ckr123tw/YOUR-NEW-REPO-NAME.git
git branch -M main
git push -u origin main
```

## Authentication

### Using Personal Access Token (Recommended)

1. Generate token:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo` (all)
   - Generate and copy token

2. When pushing, use token as password:
   ```
   Username: ckr123tw
   Password: [paste your token]
   ```

### Using GitHub CLI (Alternative)

```bash
# Install GitHub CLI
winget install --id GitHub.cli

# Authenticate
gh auth login

# Push repository
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
gh repo create rna-codon-optimizer --public --source=. --remote=origin --push
```

## Verify Upload

After pushing, visit:
https://github.com/ckr123tw/rna-codon-optimizer

You should see:
- All source files
- README.md as homepage
- GitHub will automatically detect Python project

## Next Steps

1. **Add Topics** (on GitHub web interface):
   - bioinformatics
   - machine-learning
   - rna
   - reinforcement-learning
   - deep-learning

2. **Enable GitHub Actions** (optional):
   - Add `.github/workflows/tests.yml` for automated testing

3. **Create Release**:
   ```bash
   git tag -a v1.0.0 -m "Initial release with multi-metric support"
   git push origin v1.0.0
   ```

## Troubleshooting

**Error: "git: command not found"**
- Install Git for Windows
- Restart PowerShell

**Error: "Permission denied"**
- Check your authentication token
- Ensure you have push access to the repository

**Error: "Repository not found"**
- Verify the repository exists on GitHub
- Check the remote URL: `git remote -v`
