import fire
from src.RAG_pipeline import RAGPipeline

if __name__ == "__main__":
    try:
        fire.Fire(RAGPipeline)
    except KeyboardInterrupt:
        print("\nWhen we die we go bye bye\n\t\t-Abraham Lincolin")
    except Exception as e:
        print(e)
