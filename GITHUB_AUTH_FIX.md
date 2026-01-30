# Fixing GitHub Authentication Error

## The Problem

GitHub deprecated password authentication in August 2021. You need to use a **Personal Access Token (PAT)** or **SSH keys**.

---

## ✅ SOLUTION 1: Personal Access Token (Easiest)

### Step 1: Create a Personal Access Token

1. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/tokens
   - Or: GitHub → Click your profile picture → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate New Token:**
   - Click **"Generate new token"** → **"Generate new token (classic)"**

3. **Configure Token:**
   - **Note:** `rna-codon-optimizer` (or any descriptive name)
   - **Expiration:** 90 days (or custom)
   - **Select scopes:** Check these boxes:
     - ✅ `repo` (Full control of private repositories) - **THIS IS REQUIRED**
     - ✅ `workflow` (if you plan to use GitHub Actions)

4. **Generate and Copy:**
   - Click **"Generate token"** at the bottom
   - **⚠️ COPY THE TOKEN IMMEDIATELY** - you won't see it again!
   - It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Use Token Instead of Password

When Git asks for credentials:

```
Username: ckr123tw
Password: [PASTE YOUR TOKEN HERE, NOT YOUR PASSWORD]
```

### Step 3: Store Token to Avoid Re-entering

**Option A: Git Credential Manager (Windows)**
```bash
# Store credentials (will prompt once, then remember)
git config --global credential.helper manager-core

# Now push - it will ask once and remember the token
git push -u origin main
```

**Option B: Store in Git Config (Less Secure)**
```bash
# Store credentials in plain text (only use on secure computer)
git config --global credential.helper store

# Push - will ask once and save
git push -u origin main
```

---

## ✅ SOLUTION 2: SSH Keys (More Secure, One-Time Setup)

### Step 1: Generate SSH Key

```bash
# Open PowerShell and run:
ssh-keygen -t ed25519 -C "your.email@example.com"

# Press Enter for default location
# Press Enter for no passphrase (or set one)
```

### Step 2: Copy Public Key

```bash
# Copy the public key to clipboard
cat ~/.ssh/id_ed25519.pub | clip
```

### Step 3: Add to GitHub

1. Go to: https://github.com/settings/keys
2. Click **"New SSH key"**
3. Title: `My Computer` (or any name)
4. Key: Paste the copied key
5. Click **"Add SSH key"**

### Step 4: Change Remote URL to SSH

```bash
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer

# Change from HTTPS to SSH
git remote set-url origin git@github.com:ckr123tw/rna-codon-optimizer.git

# Now push (no password needed!)
git push -u origin main
```

---

## ✅ SOLUTION 3: GitHub CLI (Most User-Friendly)

### Step 1: Install GitHub CLI

```bash
# Using winget
winget install --id GitHub.cli

# Or download from: https://cli.github.com/
```

### Step 2: Authenticate

```bash
# This will open a browser for authentication
gh auth login

# Follow prompts:
# - GitHub.com
# - HTTPS
# - Login with a web browser
```

### Step 3: Push

```bash
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
git push -u origin main
```

---

## 🚀 Quick Fix (Try This First)

**If you just want to push RIGHT NOW:**

1. **Create token:** https://github.com/settings/tokens/new
   - Select: `repo` scope
   - Generate token
   - Copy it: `ghp_xxxxx...`

2. **Push with token:**
   ```bash
   cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
   
   # When asked for password, paste the token instead
   git push -u origin main
   ```

3. **Username:** `ckr123tw`
   **Password:** `ghp_xxxxx...` (YOUR TOKEN, NOT YOUR PASSWORD)

---

## Troubleshooting

### "fatal: repository not found"
- Make sure you created the repository on GitHub first
- Check the remote URL: `git remote -v`

### Token not working
- Make sure you selected `repo` scope when creating the token
- Check token hasn't expired
- Make sure you're pasting the full token (starts with `ghp_`)

### SSH "Permission denied"
- Make sure you added the PUBLIC key (`.pub` file) to GitHub
- Test connection: `ssh -T git@github.com`

---

## Recommended Approach

**For easiest setup:** Use **Personal Access Token** with credential manager

```bash
# 1. Create token at: https://github.com/settings/tokens/new
# 2. Configure Git to remember credentials:
git config --global credential.helper manager-core

# 3. Push (will ask for token once, then remember):
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
git push -u origin main

# Username: ckr123tw
# Password: [paste your token]
```

Done! Git will remember your token for future pushes.
