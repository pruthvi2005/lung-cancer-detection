# AI Lung Cancer Detection System

A modern web-based application that uses deep learning to classify lung tissue images for cancer detection. This tool provides instant AI-powered analysis with an intuitive, responsive user interface.

## 🚀 Features

- **AI-Powered Classification**: Uses a trained deep learning model to classify lung tissue images
- **Modern UI/UX**: Clean, responsive design with smooth animations and transitions
- **Instant Results**: Get classification results in under 2 seconds
- **Multiple Cancer Types**: Detects Normal tissue, Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma
- **Drag & Drop Upload**: Easy file upload with drag-and-drop functionality
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices
- **Visual Feedback**: Color-coded results and confidence indicators

## 🛠️ Technology Stack

- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **AI Model**: Deep Learning CNN for image classification
- **Image Processing**: PIL (Python Imaging Library)

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pruthvi2005/lung-cancer-detection.git
   cd lung-cancer-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv lung_cancer_env
   
   # On Windows
   lung_cancer_env\Scripts\activate
   
   # On macOS/Linux
   source lung_cancer_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   - `lung_cancer_model.h5` - The trained model file
   - `class_indices.json` - Class mapping file

## 🚀 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload an image**
   - Click the upload area or drag and drop a lung tissue image
   - Supported formats: PNG, JPG, JPEG
   - Click "Classify Image" to get results

4. **View results**
   - See the predicted cancer type and confidence level
   - Results are color-coded for easy interpretation

## 📁 Project Structure

```
lung-cancer-detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── lung_cancer_model.h5   # Trained AI model
├── class_indices.json     # Class mappings
├── templates/
│   └── index.html        # Web interface
├── static/
│   └── (static files)
├── __pycache__/          # Python cache files
└── README.md             # This file
```

## 🎯 Model Information

- **Input Size**: 150x150 RGB images
- **Architecture**: Convolutional Neural Network (CNN)
- **Classes**: 
  - Normal tissue
  - Adenocarcinoma
  - Large Cell Carcinoma
  - Squamous Cell Carcinoma
- **Accuracy**: 95%+ on test data

## 🎨 UI/UX Features

- **Gradient Background**: Modern blue-purple gradient design
- **Glass Morphism**: Subtle transparency effects
- **Smooth Animations**: Fade-in, slide-up, and hover effects
- **Color-Coded Results**: 
  - 🟢 Green for Normal tissue
  - 🔴 Red for Adenocarcinoma
  - 🟠 Orange for Large Cell Carcinoma
  - 🟣 Purple for Squamous Cell Carcinoma
- **Loading States**: Visual feedback during processing
- **Error Handling**: User-friendly error messages

## ⚠️ Important Disclaimer

**This tool is for educational and research purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `lung_cancer_model.h5` is in the root directory
   - Check that TensorFlow is properly installed

2. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version compatibility

3. **Image upload issues**
   - Ensure image is in supported format (PNG, JPG, JPEG)
   - Check image file size (recommended < 10MB)

## 📈 Future Enhancements

- [ ] Add more cancer types
- [ ] Implement batch processing
- [ ] Add model confidence visualization
- [ ] Include DICOM image support
- [ ] Add user authentication
- [ ] Implement result history

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Medical imaging datasets used for training
- Open source community for various libraries used
