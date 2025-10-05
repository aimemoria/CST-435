# 🚨 STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS

## **IMPORTANT: Main File Configuration**

**The main application file is:** `streamlit_app.py`

### If you see an error about `create_test_video.py`:

This means Streamlit Cloud is trying to run the wrong file. Follow these steps:

## ✅ **Fix Steps:**

### **Option 1: Update Deployment Settings (Recommended)**

1. Go to your Streamlit Cloud app dashboard
2. Click on the **three dots (⋮)** next to your app
3. Select **"Settings"**
4. Under **"Main file path"**, change it to: `streamlit_app.py`
5. Click **"Save"**
6. Click **"Reboot app"**

### **Option 2: Manual Configuration**

If you deployed via GitHub:

1. Go to https://share.streamlit.io/
2. Find your app
3. Click **"⋮ More options"** → **"Settings"**
4. Set **Main file path** to: `streamlit_app.py`
5. **Save** and **Reboot**

### **Option 3: Redeploy from Scratch**

1. Delete the current deployment
2. Create a new app
3. **When asked for the main file, select:** `streamlit_app.py`
4. Deploy

## 📁 **Project Structure**

```
AIT-204-CNN-explosion-or-explanation/
├── streamlit_app.py          ← **THIS IS THE MAIN FILE**
├── requirements.txt
├── packages.txt
├── src/
│   ├── app.py               (for local development)
│   ├── models/
│   └── helpers/
├── weights/
│   └── best.pt
└── _create_test_video.py    (helper script, NOT the main app)
```

## ⚠️ **Common Errors & Solutions**

### Error: `ModuleNotFoundError: import cv2`
- **Cause:** `create_test_video.py` is being run instead of `streamlit_app.py`
- **Solution:** Update main file path to `streamlit_app.py`

### Error: `No module named 'models'`
- **Cause:** Python path not set correctly
- **Solution:** Use `streamlit_app.py` which has proper path setup

## 🧪 **Test Locally First**

Before deploying, test locally:

```bash
streamlit run streamlit_app.py
```

If this works locally, it will work on Streamlit Cloud (once configured correctly).

## 📝 **Deployment Checklist**

- [x] `streamlit_app.py` exists in root directory
- [x] Helper scripts renamed to `_*.py` to prevent auto-detection
- [ ] **Streamlit Cloud main file set to `streamlit_app.py`** ← **YOU NEED TO DO THIS**
- [ ] App deployed and tested

---

**Once you update the main file path to `streamlit_app.py`, your app will work correctly!** 🎉
