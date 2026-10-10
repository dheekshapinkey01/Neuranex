# NeuraNex — AI-Powered Resume Screening & Job Matching System

NeuraNex is a machine learning and Natural Language Processing (NLP) project designed to analyze resumes, extract relevant skills, predict suitable job categories, and match candidate profiles with job requirements.

The system helps identify relevant skills, evaluate job compatibility, and highlight skill gaps to support more informed career decisions.

## Features

- **Resume Parsing:** Extracts text from uploaded resumes.
- **Text Preprocessing:** Cleans and prepares resume text for analysis.
- **Skill Extraction:** Identifies relevant technical skills from resumes.
- **Job Category Prediction:** Predicts a suitable job category using a trained machine learning model.
- **Job Matching:** Compares resume content and skills against job descriptions.
- **Job Match Score:** Calculates a similarity-based score to indicate how closely a resume matches a job.
- **Skill Gap Analysis:** Identifies matching skills, missing skills, and recommended skills for improvement.

## Supported Job Categories

The system supports the following job categories:

1. AI/ML Engineer
2. Data Analyst
3. Software Developer
4. Web Developer
5. Cybersecurity Analyst
6. Cloud Engineer
7. Database Administrator
8. DevOps Engineer
9. Mobile App Developer
10. QA/Test Engineer
11. Business Analyst

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** Text preprocessing and skill extraction
- **Feature Extraction:** TF-IDF
- **Similarity Analysis:** Text similarity for job matching
- **Data Processing:** Pandas and NumPy
- **Web Interface:** Streamlit
- **Development Tools:** Visual Studio Code, Jupyter Notebook, Git and GitHub

## Project Workflow

1. Upload a resume.
2. Extract text from the resume.
3. Preprocess the extracted text.
4. Identify relevant skills.
5. Predict the job category.
6. Compare the resume with job requirements.
7. Calculate a job match score.
8. Display matching skills, missing skills, and recommended skills.

## Project Structure

```text
NeuraNex/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── job_category_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 00_create_dataset.ipynb
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_skill_extraction.ipynb
│   ├── 06_job_matching.ipynb
│   └── 07_resume_parser.ipynb
├── src/
│   ├── job_matcher.py
│   ├── resume_parser.py
│   ├── skill_extractor.py
│   └── text_preprocessing.py
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/dheekshapinkey01/Neuranex.git
cd Neuranex
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

Install the packages required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed in the terminal to access the application.

## Machine Learning Approach

NeuraNex combines text processing, skill extraction, machine learning-based job classification, and TF-IDF-based text similarity to analyze resumes and compare them with job requirements.

The system generates results such as:

- Predicted job category
- Extracted resume skills
- Job match score
- Matching skills
- Missing skills
- Recommended skills

## Learning Outcomes

Through this project, I gained practical experience in:

- Developing an end-to-end machine learning application
- Applying NLP techniques to resume text
- Building a resume parsing and skill extraction pipeline
- Working with TF-IDF and text similarity
- Implementing job category prediction
- Integrating machine learning functionality into a Streamlit application
- Managing project files and version control using Git and GitHub

## Developer

**Dheeksha M**

Independently designed and developed NeuraNex, from data preprocessing and model development to resume analysis, job matching, and application integration.

## Future Enhancements

- Improve resume parsing across different formats.
- Enhance skill extraction and job matching.
- Add resume analysis reports and downloadable results.
- Improve the user interface and deployment experience.
- Expand job descriptions and skill requirements.

---

*NeuraNex — Connecting Skills with Career Opportunities.*