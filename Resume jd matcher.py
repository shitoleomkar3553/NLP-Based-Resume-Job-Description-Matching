import re, math
from collections import Counter
from pypdf import PdfReader


def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

SKILLS = ["python", "java", "javascript", "react", "node.js", "sql", "mongodb",
          "html", "css", "machine learning", "docker", "aws", "git", "nlp",
          "kubernetes", "communication", "teamwork", "rest api"]

STOPWORDS = set("a an the is are was were of in on for to and with your".split())


def clean(text):
    text = re.sub(r"[^a-z0-9+.\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [w for w in clean(text).split() if w not in STOPWORDS and len(w) > 1]


def extract_skills(text):
    text = " " + clean(text) + " "
    return {s for s in SKILLS if re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", text)}


def extract_experience(text):
    nums = re.findall(r"(\d+)\+?\s*years?", text.lower())
    return max(map(int, nums)) if nums else 0


def tfidf_similarity(doc1, doc2):
    t1, t2 = tokenize(doc1), tokenize(doc2)
    vocab = set(t1) | set(t2)
    idf = {w: math.log(3 / (1 + (w in t1) + (w in t2))) + 1 for w in vocab}

    def vec(tokens):
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {w: (tf[w] / n) * idf[w] for w in vocab}

    v1, v2 = vec(t1), vec(t2)
    dot = sum(v1[w] * v2[w] for w in vocab)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    return (dot / (n1 * n2)) if n1 and n2 else 0.0


def match(resume, jd):
    sim = round(tfidf_similarity(resume, jd) * 100, 2)
    r_skills, j_skills = extract_skills(resume), extract_skills(jd)
    matched = sorted(r_skills & j_skills)
    missing = sorted(j_skills - r_skills)
    coverage = round(len(matched) / len(j_skills) * 100, 2) if j_skills else 0
    final = round(sim * 0.6 + coverage * 0.4, 2)
    return {
        "similarity_score": sim,
        "skill_coverage": coverage,
        "final_match_score": final,
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_experience": extract_experience(resume),
        "jd_required_experience": extract_experience(jd),
    }


RESUME = read_pdf(r"C:\Users\Omkar\Downloads\OMKAR_SHITOLE_RESUME_OS_ (1) (1).pdf")

JD = """
We are looking for a Software Developer skilled in Python, Java, JavaScript,
HTML, CSS, SQL, MongoDB, Git and Node.js. Experience with NLP-based projects
is a strong plus. The ideal candidate should be comfortable working in a
collaborative, fast-paced development environment.
"""

if __name__ == "__main__":
    result = match(RESUME, JD)
    for k, v in result.items():
        print(f"{k}: {v}")