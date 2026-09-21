"""Repository entrypoint; run with the existing model environment on Kaggle."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from web_agent.eval.task1.cli import main
if __name__=='__main__': main()
