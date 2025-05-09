import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=True)
    return process.returncode == 0

def main():
    # Change directory
    # os.chdir("kg-counter-narratives")
    
    # Set up conda environment
    # run_command("conda create --name kg-counter-narratives python=3.8 -y")
    
    # This is tricky in a script - in reality, this activates the environment in a subshell
    # For a script, we'd want to use the conda Python directly instead
    conda_prefix = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'kg-counter-narratives', 'bin')
    python_path = os.path.join(conda_prefix, 'python')
    pip_path = os.path.join(conda_prefix, 'pip')
    
    # Install packages from requirements.txt
    run_command(f"{pip_path} install -r requirements.txt")
    
    # Clone and install GitHub repositories
    repos = [
        "https://github.com/leolani/cltl-knowledgeextraction",
        "https://github.com/leolani/cltl-knowledgerepresentation",
        "https://github.com/leolani/cltl-combot",
        "https://github.com/leolani/cltl-knowledgelinking",
        # "https://github.com/huggingface/neuralcoref.git"
    ]
    
    for repo in repos:
        repo_name = repo.split("/")[-1].replace(".git", "")
        run_command(f"git clone {repo}")
        os.chdir(repo_name)
        
        # Special case for neuralcoref
        if repo_name == "neuralcoref":
            run_command(f"{pip_path} install -r requirements.txt")
            run_command(f"{pip_path} install cython==0.29 --upgrade")
        
        run_command(f"{pip_path} install -e .")
        os.chdir("..")
    
    # Final installations
    run_command(f"{pip_path} install stanford-openie")
    run_command(f"{pip_path} install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz")
    
    # Add spaCy model download directly
    run_command(f"{python_path} -m spacy download en_core_web_sm")
    
    print("Installation complete!")

if __name__ == "__main__":
    main() 