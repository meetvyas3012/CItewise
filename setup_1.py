"""
Setup configuration for Multi-Document Research Assistant
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-doc-rag",
    version="1.0.0",
    author="Multi-Document RAG Team", 
    author_email="team@multi-doc-rag.com",
    description="AI-powered research assistant for multi-document analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multi-doc-rag/multi-doc-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
        "full": [
            "elasticsearch>=8.8.0",
            "redis>=4.6.0",
            "celery>=5.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "multi-doc-rag=multi_doc_rag.cli:main",
            "rag-server=multi_doc_rag.ui.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multi_doc_rag": [
            "config/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/multi-doc-rag/multi-doc-rag/issues",
        "Source": "https://github.com/multi-doc-rag/multi-doc-rag",
        "Documentation": "https://multi-doc-rag.readthedocs.io/",
    },
)
