# Health Disease Prediction System

## Overview 🌟
A web-based application using machine learning to predict potential health conditions based on patient data. Built with Streamlit and Random Forest Classifier.

## Live Demo 🚀
[View Live Demo](#) <!-- Add your deployed app link -->

## Features
- Real-time disease prediction
- Interactive web interface
- Probability breakdown
- Health recommendations
- Dark mode UI
- Mobile responsive design

## Tech Stack
- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- Random Forest Classifier

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Pritam-Shome/Health_prediction_System.git
cd Health_prediction_System
```

2. **Set up virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model**
```bash
cd scripts
python train_model.py
```

2. **Launch the application**
```bash
streamlit run app.py
```

3. Access the web interface at `http://localhost:8501`

## Input Parameters

| Parameter | Range/Values |
|-----------|-------------|
| Age | 0-120 years |
| BMI | 10.0-50.0 |
| Blood Pressure | 80-200 mmHg |
| Sugar | 50-300 mg/dL |
| Cholesterol | 100-400 mg/dL |
| Gender | Male/Female |
| Smoking | Yes/No |
| Family History | Yes/No |

## Directory Structure
```
Health_prediction_System/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── data/
│   └── health_data.csv   # Training dataset
├── models/               # Saved model files
│   ├── random_forest_model.pkl
│   └── label_encoders.pkl
└── scripts/
    └── train_model.py    # Model training script
```

## Model Details
- Algorithm: Random Forest Classifier
- Input Features: 8 parameters
- Output: Disease prediction with confidence score
- Storage: Pickle serialization



## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Your Name
- GitHub: [Pritam Shome](https://github.com/yourusername)
- Email: pritamshome1@gmail.com



---
⭐ Found this useful? Star this repository!