from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import matplotlib.pyplot as plt
import base64
import io
import json
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class ExperimentConfig(BaseModel):
    labels: List[str]
    title: str

class ExperimentRequest(BaseModel):
    experiments: List[ExperimentConfig]

class PoolData(BaseModel):
    file_path: str
    total: int
    unique: int
    overlapping: int
    unique_fraction: float
    ci: Tuple[float, float]

class Comparison(BaseModel):
    pool_1: str
    pool_2: str
    rd: float
    ci: Tuple[float, float]
    p_value: float
    z_stat: float

class ExperimentResult(BaseModel):
    title: str
    data: List[PoolData]
    comparisons: List[Comparison]

class AnalysisResponse(BaseModel):
    results: List[ExperimentResult]
    plot: str  # Base64 encoded plot

def get_idea_embeddings(ideas: List[str]) -> np.ndarray:
    return embed(ideas)

def calculate_unique_ideas(ideas: List[str], threshold: float = 0.80) -> Tuple[int, int, int]:
    vectors = get_idea_embeddings(ideas)
    similar_ideas_count = 0

    for i, vector in enumerate(vectors):
        if i == 0:
            continue
        previous_vectors = vectors[:i]
        similarity_scores = cosine_similarity([vector], previous_vectors)
        if np.max(similarity_scores) >= threshold:
            similar_ideas_count += 1
    
    total = len(ideas)
    unique_count = total - similar_ideas_count
    return total, unique_count, similar_ideas_count

def create_analysis_plot(experiment_results: List[ExperimentResult]) -> str:
    COLOR_MAP = {
        "ChatGPT-assisted": "#555555",
        "Websearch-assisted": "#AAAAAA",
        "Human": "#333333",
        "ChatGPT-Human": "#555555",
        "ChatGPT-Only": "#888888",
        "Human-Only": "#333333",
        "ChatGPT (high constraints)": "#555555",
        "ChatGPT (low constraints)": "#777777",
        "Websearch (high constraints)": "#AAAAAA",
        "Websearch (low constraints)": "#CCCCCC",
        "ChatGPT Empathy Baseline": "#555555",
        "ChatGPT Empathy High": "#777777",
        "Websearch Empathy Baseline": "#AAAAAA",
        "Websearch Empathy High": "#CCCCCC"
    }

    fig, axes = plt.subplots(3, 2, figsize=(22, 18))
    axes = axes.flatten()

    for i, experiment in enumerate(experiment_results):
        exp_data = experiment.data
        labels = [data.file_path for data in exp_data]
        proportions = [data.unique_fraction for data in exp_data]
        colors = [COLOR_MAP.get(label, "#888888") for label in labels]

        x = np.arange(len(labels))
        axes[i].bar(x, proportions, color=colors)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].set_ylabel("Unique Fraction")
        axes[i].set_title(experiment.title)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return plot_base64

async def process_file(file: UploadFile) -> List[str]:
    content = await file.read()
    text = content.decode('utf-8')
    return [line.strip() for line in text.split('\n') if line.strip()]

@app.post("/analyze")
async def analyze_experiments(
    files: List[UploadFile] = File(...),
    config: str = Form(...)
) -> Dict[str, Any]:
    try:
        # Parse the JSON config string
        experiment_configs = json.loads(config)
        results = []
        
        # Group files by experiment
        file_index = 0
        for exp_config in experiment_configs['experiments']:
            exp_data = []
            n_files = len(exp_config['labels'])
            
            # Process files for this experiment
            for label in exp_config['labels']:
                if file_index >= len(files):
                    raise HTTPException(
                        status_code=400,
                        detail="Number of files doesn't match configuration"
                    )
                
                try:
                    # Read and process the file
                    ideas = await process_file(files[file_index])
                    file_index += 1
                    
                    total, unique, overlapping = calculate_unique_ideas(ideas)
                    unique_fraction = unique / total
                    ci = proportion_confint(count=unique, nobs=total, alpha=0.05, method='beta')
                    
                    exp_data.append(PoolData(
                        file_path=label,
                        total=total,
                        unique=unique,
                        overlapping=overlapping,
                        unique_fraction=unique_fraction,
                        ci=ci
                    ))
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing file for {label}: {str(e)}"
                    )

            # Calculate comparisons
            comparisons = []
            for i, pool_1 in enumerate(exp_data):
                for j, pool_2 in enumerate(exp_data):
                    if i >= j:
                        continue
                    
                    successes = [pool_1.unique, pool_2.unique]
                    nobs = [pool_1.total, pool_2.total]
                    z_stat, p_value = proportions_ztest(successes, nobs)
                    rd = pool_1.unique_fraction - pool_2.unique_fraction
                    
                    n_comparisons = len(exp_config['labels']) * (len(exp_config['labels']) - 1) / 2
                    p_value = min(p_value * n_comparisons, 1.0)
                    
                    comparisons.append(Comparison(
                        pool_1=pool_1.file_path,
                        pool_2=pool_2.file_path,
                        rd=rd,
                        ci=(rd - 1.96 * np.sqrt(rd * (1-rd) / pool_1.total),
                            rd + 1.96 * np.sqrt(rd * (1-rd) / pool_1.total)),
                        p_value=p_value,
                        z_stat=z_stat
                    ))
            
            results.append(ExperimentResult(
                title=exp_config['title'],
                data=exp_data,
                comparisons=comparisons
            ))
        
        plot_base64 = create_analysis_plot(results)
        
        return {
            "results": [result.dict() for result in results],
            "plot": plot_base64
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))