# GitHub License Setup for CC BY-NC 4.0

## The Issue

GitHub's default license chooser doesn't include Creative Commons licenses like CC BY-NC 4.0.

## Solution: Manual LICENSE File (Already Done! ✅)

We've already created the LICENSE file with CC BY-NC 4.0 text. When you push to GitHub:

1. **GitHub will automatically detect it** from the file content
2. The repository will show "CC-BY-NC-4.0" in the license badge
3. No additional action needed!

## Steps When Creating Repository on GitHub

### Option 1: Create Repo WITHOUT License (Recommended)

1. Go to https://github.com/new
2. Repository name: `rna-codon-optimizer`
3. **License: "None"** (leave it empty)
4. Create repository
5. Push your code (which includes our LICENSE file)
6. GitHub will automatically recognize CC BY-NC 4.0

```bash
cd C:\Users\ckr12\.gemini\antigravity\scratch\rna-codon-optimizer
git init
git add .
git commit -m "Initial commit: RNA codon optimizer with CC BY-NC 4.0 license"
git remote add origin https://github.com/ckr123tw/rna-codon-optimizer.git
git branch -M main
git push -u origin main
```

### Option 2: Add License Badge Manually

GitHub might not automatically detect CC licenses. Add this to your README (already done):

```markdown
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
```

### Option 3: Alternative - Use AGPL-3.0 with Commercial Exception

If you want a license in GitHub's chooser that's close to "academic only":

**AGPL-3.0** (with custom commercial exception in header)
- More restrictive than CC BY-NC
- Requires derivative works to share source code
- In GitHub's license picker
- Add commercial exception clause

**Not recommended** because it's more complex than CC BY-NC 4.0.

## Verification After Push

After pushing to GitHub:

1. Go to your repository: `https://github.com/ckr123tw/rna-codon-optimizer`
2. Check the "About" section (right sidebar)
3. It should show the license type
4. If it doesn't auto-detect:
   - Click "⚙️" next to "About"
   - Manually select "CC-BY-NC-4.0" if available
   - Or add it in the description

## Add LICENSE Badge to Repository

To make the license super clear, add a file `.github/FUNDING.yml` (optional) or just ensure README.md prominently displays the license.

We've already added the license badge to README.md:
```markdown
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
```

## Important Note

The LICENSE file we created is **legally binding**. GitHub's license detection is just for convenience. Your repository is properly licensed even if GitHub doesn't detect it automatically.

## Summary

✅ **You don't need to do anything special!**

1. Create GitHub repo with "No license" option
2. Push your code (includes LICENSE file)
3. GitHub will recognize it

The LICENSE file is already in your project and will work perfectly when pushed to GitHub.
