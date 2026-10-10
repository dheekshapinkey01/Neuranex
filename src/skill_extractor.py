import re


def extract_skills(text, skill_dictionary):
    """Extract known skills from resume text."""

    text = text.lower()

    extracted_skills = []

    for category_skills in skill_dictionary.values():

        for skill in category_skills:

            pattern = r"\b" + re.escape(skill.lower()) + r"\b"

            if re.search(pattern, text):
                extracted_skills.append(skill)

    return list(dict.fromkeys(extracted_skills))