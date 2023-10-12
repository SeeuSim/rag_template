from pathlib import Path
from ray.data import from_items


def get_items():
    DOCS_DIR = Path(
        './documents', 
    )
    for doc in DOCS_DIR.rglob('.pdf'):
        print(doc)

if __name__ == '__main__':
    get_items()