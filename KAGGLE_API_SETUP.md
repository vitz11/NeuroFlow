# 🔐 Kaggle API Setup - Two Methods

## Method 1: Using Environment Variables (Recommended - What You Got)

If Kaggle gave you code like this:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Setup on Windows PowerShell:

**Option A: Temporary (Current Session Only)**
```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_api_key"
```

**Option B: Permanent (All Future Sessions)**
```powershell
# Open PowerShell as Administrator, then run:
[Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
[Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_api_key", "User")

# Then restart PowerShell
```

**Option C: Using .env File (Easiest)**

1. Create `.env` file in your project root:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. Add this to your code (at the start):
```python
import os
from dotenv import load_dotenv

load_dotenv()  # This loads from .env file
```

---

## Method 2: Create kaggle.json File Manually

If you want the traditional `kaggle.json` file instead:

### Step 1: Create the Directory
```powershell
mkdir "$env:USERPROFILE\.kaggle"
```

### Step 2: Create kaggle.json File

Create a file at: `C:\Users\vitth\.kaggle\kaggle.json`

With this content (replace with YOUR credentials):
```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

### Step 3: Verify It Works
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("✓ Kaggle API authenticated successfully!")
```

---

## 🧪 Test Your Setup

Run this script to verify everything works:

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi

print("Testing Kaggle API Setup...")
print("-" * 50)

try:
    api = KaggleApi()
    api.authenticate()
    print("✓ Successfully authenticated with Kaggle API!")
    
    # Try searching for a dataset
    print("\nSearching for 'iris' datasets...")
    datasets = api.dataset_list(search='iris')[:3]
    
    if datasets:
        print(f"✓ Found {len(datasets)} datasets:")
        for ds in datasets:
            print(f"  - {ds.ref}")
    else:
        print("✓ Connection successful (no datasets returned for this query)")
        
except Exception as e:
    print(f"✗ Error: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check your credentials are correct")
    print("2. Make sure environment variables are set")
    print("3. Or place kaggle.json at C:\\Users\\vitth\\.kaggle\\kaggle.json")
```

---

## ✅ Quick Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Environment Variables** | No file to manage, Secure | Must set each time (or setup permanently) | CI/CD, Shared computers |
| **kaggle.json File** | Works automatically, Simple | File must stay private, Don't commit | Local development |
| **.env File** | Clean, Organized, Local only | Need python-dotenv package | Development teams |

---

## 🎯 I Recommend for You:

### Use the .env File Method:

1. **Install python-dotenv:**
```bash
pip install python-dotenv
```

2. **Create `.env` file in your project root:**
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

3. **Add to `.gitignore`** (already there):
```
.env
```

4. **Update your code** to load it:
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file
```

This way:
- ✅ Credentials stay private (in .gitignore)
- ✅ Works automatically
- ✅ Easy to share project without sharing credentials
- ✅ No need to set environment variables manually

---

## 📋 Step-by-Step for Your Setup

1. **Get your credentials from Kaggle:**
   - Go to https://www.kaggle.com/settings/account
   - You already have: `KAGGLE_USERNAME` and `KAGGLE_KEY`

2. **Choose your method** (I recommend .env):
   - Option A: Set environment variables permanently
   - Option B: Create kaggle.json file
   - Option C: Use .env file (easiest)

3. **Test it:**
   - Run the test script above
   - Should see: `✓ Successfully authenticated with Kaggle API!`

4. **You're ready!**
   - Use in your code
   - Download datasets
   - Run your AutoML system

---

## 🆘 If It's Still Not Working

Try this diagnostic script:

```python
import os
import sys

print("Kaggle Setup Diagnostic")
print("=" * 50)

# Check environment variables
print("\n1. Checking environment variables:")
username = os.getenv('KAGGLE_USERNAME')
key = os.getenv('KAGGLE_KEY')

if username and key:
    print(f"   ✓ KAGGLE_USERNAME: {username[:5]}...")
    print(f"   ✓ KAGGLE_KEY: {key[:10]}...")
else:
    print(f"   ✗ KAGGLE_USERNAME: Not set")
    print(f"   ✗ KAGGLE_KEY: Not set")

# Check kaggle.json file
print("\n2. Checking kaggle.json file:")
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
if os.path.exists(kaggle_json):
    print(f"   ✓ Found at: {kaggle_json}")
else:
    print(f"   ✗ Not found at: {kaggle_json}")

# Try authentication
print("\n3. Testing authentication:")
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("   ✓ Authentication successful!")
except Exception as e:
    print(f"   ✗ Authentication failed: {str(e)}")

print("\n" + "=" * 50)
```

---

**Let me know which method you want to use and I'll help you set it up!** 🚀
