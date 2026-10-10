import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_skills(text, job_skills):
    """Extract known skills from resume text."""

    text = text.lower()
    extracted_skills = []

    for category_skills in job_skills.values():
        for skill in category_skills:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"

            if re.search(pattern, text):
                extracted_skills.append(skill)

    return list(dict.fromkeys(extracted_skills))


def compare_skills(resume_skills, required_skills):
    """Find matching and missing skills."""

    resume_skills_lower = {
        skill.lower().strip()
        for skill in resume_skills
    }

    required_skills_list = [
        skill.strip()
        for skill in required_skills.split(",")
    ]

    matching_skills = []
    missing_skills = []

    for skill in required_skills_list:
        if skill.lower() in resume_skills_lower:
            matching_skills.append(skill)
        else:
            missing_skills.append(skill)

    return matching_skills, missing_skills


def calculate_job_matches(resume_text, jobs_df):
    """Calculate resume similarity against all jobs."""

    job_text = (
        jobs_df["job_title"].fillna("") + " " +
        jobs_df["job_description"].fillna("") + " " +
        jobs_df["required_skills"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    job_vectors = vectorizer.fit_transform(job_text)

    resume_vector = vectorizer.transform([resume_text])

    similarity_scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    results = pd.DataFrame({
        "Job Title": jobs_df["job_title"],
        "Match Score": similarity_scores * 100
    })

    results = results.sort_values(
        by="Match Score",
        ascending=False
    ).reset_index(drop=True)

    results["Match Score"] = results["Match Score"].round(2)

    return results

def recommend_skills(missing_skills, max_recommendations=5):
    """Recommend skills based on missing job requirements."""

    return missing_skills[:max_recommendations]