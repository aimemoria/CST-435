# Streamlit Cloud Deployment Guide

## ✅ Files Created for Deployment

The following files have been created to ensure smooth deployment on Streamlit Cloud:

1. **`streamlit_app.py`** - Main app file (Streamlit Cloud looks for this)
2. **`packages.txt`** - System-level dependencies (ffmpeg, OpenCV libraries)
3. **`.streamlit/config.toml`** - Streamlit configuration
4. **`.gitignore`** - Prevents helper scripts from being run as the main app

## 🚀 Deployment Steps for Streamlit Cloud

### Step 1: Push to GitHub

```bash
cd AIT-204-CNN-explosion-or-explanation
git init
git add .
git commit -m "Initial commit: CNN Explosion or Explanation classifier"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. **Main file path:** Select `streamlit_app.py` (NOT `create_test_video.py`)
5. Click "Deploy"

### Step 3: Configure Advanced Settings (if needed)

In Streamlit Cloud advanced settings:
- **Python version:** 3.11 or 3.12
- **Main file:** `streamlit_app.py`

## 📁 Required Files on GitHub

Make sure these files are in your repository:

```
AIT-204-CNN-explosion-or-explanation/
├── streamlit_app.py          # ← Main entry point
├── requirements.txt           # Python dependencies
├── packages.txt              # System dependencies
├── .streamlit/
│   └── config.toml           # Streamlit config
├── src/
│   ├── app.py                # Original app (kept for local use)
│   ├── models/
│   │   └── explex_net.py
│   └── helpers/
│       ├── decode_video.py
│       ├── audio.py
│       ├── windows.py
│       └── datasets.py
├── weights/
│   └── best.pt               # Trained model weights
└── data/
    └── sample_annotations/
```

## ⚠️ Important Notes

### Model Weights
The `weights/best.pt` file is **76MB** which is within GitHub's file size limit (100MB). However:
- If deployment fails due to file size, you can use Git LFS (Large File Storage)
- Or host weights externally (Google Drive, Hugging Face) and download them in the app

### Video Files
The test video (`videos/test_science_demo.mp4`) is excluded from deployment via `.gitignore` because:
- It's not needed on Streamlit Cloud
- Users will upload their own videos
- Reduces deployment size

### System Dependencies
`packages.txt` ensures OpenCV and ffmpeg work correctly on Streamlit Cloud's Linux environment.

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'cv2'"
**Solution:** Ensure `packages.txt` contains:
```
ffmpeg
libsm6
libxext6
```

### Error: "This app has encountered an error"
**Solution:**
1. Check "Manage app" → "Logs" in Streamlit Cloud
2. Verify the main file is set to `streamlit_app.py`
3. Ensure all imports in `streamlit_app.py` have the correct path

### Error: "Model weights not found"
**Expected behavior:** The app will show a warning but use a randomly initialized model. Users can still test the interface.

**To fix:** Ensure `weights/best.pt` is committed to GitHub:
```bash
git add weights/best.pt
git commit -m "Add trained model weights"
git push
```

## 🧪 Testing Deployment Locally

Before deploying to Streamlit Cloud, test with the deployment file locally:

```bash
streamlit run streamlit_app.py
```

This should work exactly like `src/app.py`.

## 📊 Expected Performance on Streamlit Cloud

- **Cold start:** 30-60 seconds (installing dependencies)
- **Warm start:** 2-5 seconds
- **Video processing:** Depends on video length and parameters
  - 30-second video: ~10-20 seconds
  - 60-second video: ~20-40 seconds

## 🎯 Post-Deployment Testing

Once deployed:
1. Upload the test video (if you want, add it to the repo without gitignore)
2. Try different parameter combinations
3. Check that the probability chart displays correctly
4. Verify explosion detection alerts appear

## 📝 Deployment Checklist

- [x] `streamlit_app.py` created in root directory
- [x] `packages.txt` added for system dependencies
- [x] `.streamlit/config.toml` configured
- [x] `.gitignore` prevents helper scripts from being main app
- [x] `requirements.txt` has all Python dependencies
- [x] `weights/best.pt` is committed (if < 100MB)
- [ ] Repository pushed to GitHub
- [ ] Streamlit Cloud deployment configured
- [ ] Main file set to `streamlit_app.py`
- [ ] Deployment tested and working

---

**Your app is ready for deployment! Follow the steps above to deploy to Streamlit Cloud.** 🚀
