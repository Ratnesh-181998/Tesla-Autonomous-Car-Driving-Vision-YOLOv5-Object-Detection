# GitHub Upload Guide - Tesla Autonomous Vision Project

This guide will help you upload the project to GitHub with Git LFS support for large files.

## Prerequisites

1. **Git** - [Download Git](https://git-scm.com/downloads)
2. **Git LFS** - [Download Git LFS](https://git-lfs.github.com/)
3. **GitHub Account** - [Sign up](https://github.com/signup)

## Step-by-Step Upload Process

### Step 1: Install Git LFS

```bash
# Windows (using Git Bash or PowerShell)
git lfs install

# Verify installation
git lfs version
```

### Step 2: Navigate to Project Directory

```bash
cd "C:\Users\rattu\Downloads\L8 P-2 ObjectDetection with Single Stage Methods\Project Run -1"
```

### Step 3: Initialize Git Repository

```bash
# Initialize git repository
git init

# Configure user (replace with your details)
git config user.name "Ratnesh Singh"
git config user.email "rattudacsit2021gate@gmail.com"
```

### Step 4: Track Large Files with Git LFS

```bash
# The .gitattributes file is already configured
# Verify LFS tracking
git lfs track

# You should see:
# *.onnx
# *.mp4
# *.pdf
# *.zip
# etc.
```

### Step 5: Add Remote Repository

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/Ratnesh-181998/Tesla-Autonomous-Car-Driving-Vision-YOLOv5-Object-Detection.git

# Verify remote
git remote -v
```

### Step 6: Stage All Files

```bash
# Add all files to staging
git add .

# Check status
git status
```

### Step 7: Commit Changes

```bash
# Create initial commit
git commit -m "Initial commit: Tesla Autonomous Vision - YOLOv5 Object Detection

- Complete Streamlit application with Tesla-inspired UI
- YOLOv5 ONNX models (nano and small variants)
- Real-time object detection for autonomous driving
- Comprehensive documentation and README
- Demo video and sample outputs
- MIT License"
```

### Step 8: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main

# If you encounter authentication issues, use Personal Access Token
# GitHub Settings > Developer settings > Personal access tokens > Generate new token
```

### Step 9: Verify Upload

1. Go to: https://github.com/Ratnesh-181998/Tesla-Autonomous-Car-Driving-Vision-YOLOv5-Object-Detection
2. Check that all files are uploaded
3. Verify LFS files show correct sizes
4. Ensure README.md displays correctly

## Large Files Tracked by Git LFS

The following files will be automatically tracked by Git LFS:

| File | Size | Type |
|------|------|------|
| yolov5s.onnx | 29.3 MB | Model |
| yolov5n.onnx | 7.9 MB | Model |
| Vid_Self-Driving_Demo.mp4 | 41.3 MB | Video |
| YoloV5_ONNX.zip | 31.3 MB | Archive |
| Object Detection SSM Tesla Driving Car.pdf | 12.2 MB | Documentation |
| tesla_output.mp4 | 9.4 MB | Output |
| OLD_L8_ObjectDetection_with_Single_Stage_Methods.ipynb | 4.7 MB | Notebook |

## Troubleshooting

### Issue: "This exceeds GitHub's file size limit of 100 MB"

**Solution**: Ensure Git LFS is installed and tracking is configured
```bash
git lfs install
git lfs track "*.onnx"
git lfs track "*.mp4"
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Issue: "Authentication failed"

**Solution**: Use Personal Access Token instead of password
1. GitHub Settings > Developer settings > Personal access tokens
2. Generate new token (classic)
3. Select scopes: repo, workflow
4. Copy token
5. Use token as password when pushing

### Issue: "LFS files not uploading"

**Solution**: Ensure LFS is initialized
```bash
git lfs install
git lfs migrate import --include="*.onnx,*.mp4,*.pdf,*.zip"
git push origin main --force
```

## Streamlit Cloud Deployment

### Step 1: Prepare Repository

Ensure these files exist:
- ✅ requirements.txt
- ✅ project_1_streamlit_app.py
- ✅ tesla_object_detection.py
- ✅ All model files (via Git LFS)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository**: Ratnesh-181998/Tesla-Autonomous-Car-Driving-Vision-YOLOv5-Object-Detection
   - **Branch**: main
   - **Main file path**: project_1_streamlit_app.py
5. Click "Deploy!"

### Step 3: Configure Advanced Settings (Optional)

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#E82127"
backgroundColor = "#0f1c15"
secondaryBackgroundColor = "#171A20"
textColor = "#FFFFFF"

[server]
maxUploadSize = 200
enableCORS = false
```

### Step 4: Monitor Deployment

- Check build logs for errors
- Verify all dependencies install correctly
- Test application functionality
- Update app URL in README.md

## Post-Upload Checklist

- [ ] All files uploaded successfully
- [ ] LFS files show correct sizes
- [ ] README.md displays properly
- [ ] License file present
- [ ] .gitignore working correctly
- [ ] Repository description added
- [ ] Topics/tags added
- [ ] Repository is public
- [ ] Streamlit app deployed
- [ ] App URL updated in README
- [ ] Social media preview image set

## Repository Settings

### Description
```
Real-time object detection for autonomous driving using YOLOv5 and ONNX. Tesla-inspired Streamlit UI with glassmorphic design. Detects vehicles, pedestrians, and traffic signals.
```

### Topics
```
object-detection, yolov5, onnx, streamlit, autonomous-driving, 
computer-vision, deep-learning, real-time-detection, opencv, 
pytorch, tesla, pedestrian-detection, vehicle-detection, 
traffic-signal-detection, single-stage-detector
```

### Website
```
https://universal-pdf-rag-chatbot-mhsi4ygebe6hmq3ij6d665.streamlit.app/
```

## Maintenance

### Updating the Repository

```bash
# Make changes to files
# ...

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Update: Description of changes"

# Push to GitHub
git push origin main
```

### Adding New Large Files

```bash
# Track new file type
git lfs track "*.newtype"

# Add .gitattributes
git add .gitattributes

# Add and commit new files
git add new_large_file.newtype
git commit -m "Add new large file"
git push origin main
```

## Support

If you encounter issues:
1. Check [GitHub LFS Documentation](https://git-lfs.github.com/)
2. Review [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
3. Open an issue on GitHub
4. Contact: rattudacsit2021gate@gmail.com

---

**Good luck with your deployment! 🚀**
