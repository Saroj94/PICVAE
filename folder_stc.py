import os
from pathlib import Path

project_source_name='src'

Files_Folder_list=[
    f"{project_source_name}/__init__.py",
    f"{project_source_name}/config.py",
    f"{project_source_name}/constants.py",
    f"{project_source_name}/exceptions.py",
    f"{project_source_name}/logger.py",
    f"{project_source_name}/cli.py",
    f"{project_source_name}/components/__init__.py",
    f"{project_source_name}/components/data_ingestion.py",  
    f"{project_source_name}/components/data_validation.py",
    f"{project_source_name}/components/data_transformation.py",
    f"{project_source_name}/components/model_trainer.py",
    f"{project_source_name}/components/model_evaluation.py",
    f"{project_source_name}/components/model_pusher.py",
    f"{project_source_name}/models/__init__.py",
    f"{project_source_name}/models/architecture.py",
    f"{project_source_name}/models/estimator.py",
    f"{project_source_name}/configuration/aws_connection.py",
    f"{project_source_name}/cloud_storage/__init__.py",
    f"{project_source_name}/cloud_storage/aws_storage.py",
    f"{project_source_name}/data_access/__init__.py",
    f"{project_source_name}/data_access/ivim_data.py",
    f"{project_source_name}/entity/__init__.py",
    f"{project_source_name}/entity/config_entity.py",
    f"{project_source_name}/entity/artifact_entity.py",
    f"{project_source_name}/pipeline/__init__.py",
    f"{project_source_name}/pipeline/training_pipeline.py",
    f"{project_source_name}/pipeline/prediction_pipeline.py",
    f"{project_source_name}/serving/__init__.py",
    f"{project_source_name}/serving/api.py",
    f"{project_source_name}/serving/schemas.py",
    f"{project_source_name}/utils/__init__.py",
    f"{project_source_name}/utils/io.py",
    f"{project_source_name}/utils/reproducibility.py",
    "app.py",
    "README.md",
    "requirements.txt",
    "uv.lock",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".github/workflows/ci.yaml",
    ".github/workflows/deploy.yaml",
    ".pre-commit-config.yaml",
    "notebooks/test.ipynb",
    "pyproject.toml",
    "figures/output",
    "configs/model.yaml",
    "configs/schema.yaml",
    "configs/training.yaml"
]


for filepath in Files_Folder_list:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")
