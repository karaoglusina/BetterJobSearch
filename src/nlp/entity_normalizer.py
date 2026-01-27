"""Entity normalization using rapidfuzz for deduplication."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# Canonical mappings for common abbreviations/aliases
CANONICAL_MAP: Dict[str, str] = {
    # Programming languages
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python": "Python",
    "c#": "C#",
    "csharp": "C#",
    "c++": "C++",
    "cpp": "C++",
    "golang": "Go",
    "go": "Go",
    "rb": "Ruby",
    "ruby": "Ruby",
    # ML/AI
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "large language models": "Large Language Models",
    "gen ai": "Generative AI",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    # Cloud
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "azure": "Azure",
    "microsoft azure": "Azure",
    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "ms sql": "SQL Server",
    "sql server": "SQL Server",
    "mssql": "SQL Server",
    "elasticsearch": "Elasticsearch",
    "elastic search": "Elasticsearch",
    "es": "Elasticsearch",
    "redis": "Redis",
    # Tools
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "tf": "Terraform",
    "terraform": "Terraform",
    "gh actions": "GitHub Actions",
    "github actions": "GitHub Actions",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    # Frameworks
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "angular": "Angular",
    "angularjs": "Angular",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "next": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    # Data tools
    "pbi": "Power BI",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "tableau": "Tableau",
    "looker": "Looker",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "sklearn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "keras": "Keras",
    "hf": "Hugging Face",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    # Data platforms
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "bigquery": "BigQuery",
    "big query": "BigQuery",
    "redshift": "Redshift",
    "airflow": "Airflow",
    "apache airflow": "Airflow",
    "dbt": "dbt",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "pyspark": "Apache Spark",
    "kafka": "Apache Kafka",
    "apache kafka": "Apache Kafka",
}


class EntityNormalizer:
    """Normalize extracted entities to canonical forms using exact lookup + fuzzy matching."""

    def __init__(self, *, fuzzy_threshold: int = 85):
        """Initialize normalizer.

        Args:
            fuzzy_threshold: Minimum fuzz ratio (0-100) for fuzzy matching.
        """
        self.canonical_map = {k.lower(): v for k, v in CANONICAL_MAP.items()}
        self.fuzzy_threshold = fuzzy_threshold
        self._canonical_values = list(set(CANONICAL_MAP.values()))

    def normalize(self, entity: str) -> str:
        """Normalize a single entity string to its canonical form."""
        if not entity:
            return entity

        key = entity.strip().lower()

        # Exact lookup
        if key in self.canonical_map:
            return self.canonical_map[key]

        # Fuzzy match against canonical values
        best = self._fuzzy_match(entity)
        if best:
            return best

        # Return original with title case if no match
        return entity

    def normalize_list(self, entities: List[str]) -> List[str]:
        """Normalize a list of entities, deduplicating after normalization."""
        seen: set[str] = set()
        result: List[str] = []
        for entity in entities:
            normalized = self.normalize(entity)
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    def _fuzzy_match(self, entity: str) -> Optional[str]:
        """Find best fuzzy match among canonical values."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return None

        best_score = 0
        best_match = None

        for canonical in self._canonical_values:
            score = fuzz.ratio(entity.lower(), canonical.lower())
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = canonical

        return best_match

    def add_canonical(self, alias: str, canonical: str) -> None:
        """Add a new alias -> canonical mapping."""
        self.canonical_map[alias.lower()] = canonical
        if canonical not in self._canonical_values:
            self._canonical_values.append(canonical)
