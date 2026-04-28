# 🧠 AI Medical Image Description Generator

An intelligent AI system that analyzes medical images and generates meaningful descriptions.  
Supports radiology scans, prescriptions, and medical reports using deep learning and OCR.

---

## 🚀 Features

- 🔍 Multi-Modal Input
  - Chest X-rays
  - Printed Prescriptions
  - Medical Reports

- 🧠 AI-Based Analysis
  - CNN-based disease detection (ResNet / DenseNet)
  - Multi-label classification

- 📝 OCR Processing
  - Extracts text from medical documents
  - Identifies medicines, dosage, and details

- 🔀 Intelligent Routing
  - Detects input type automatically
  - Routes to correct processing module

- 📊 Output
  - Top findings with confidence
  - Generated description
  - Safety disclaimer

---

## 🏗️ System Workflow

1. Image Upload  
2. Preprocessing  
3. Routing Module  
4. Analysis  
   - Radiology → CNN Model  
   - Documents → OCR  
5. Description Generation  
6. Result Display  

---

## 🧪 Model Performance

- Accuracy (Hamming Accuracy): **83–85%**
- Macro Precision: **31–35%**
- Macro Recall: **60–65%**
- Macro F1 Score: **40–45%**

---

## 🧰 Tech Stack

- Python
- Flask
- PyTorch
- OpenCV
- Tesseract OCR
- Pandas, NumPy

---

## 📂 Project Structure

AI_Medical_Project/
│
├── app/
├── data/
├── scripts/
├── run.py
├── README.md

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/AI-Medical-Image-Description-Generator.git
cd AI-Medical-Image-Description-Generator
pip install -r requirements.txt
▶️ Run Project
python run.py

Open:

http://127.0.0.1:5000
📊 Dataset
NIH Chest X-ray Dataset
Custom medical documents
⚠️ Disclaimer

This project is for academic purposes only.
It is not a medical diagnostic tool.

⭐ Support

If you like this project, give it a ⭐ on GitHub!