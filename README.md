# Fashion Recommender System (Lumina)

A multimodal fashion search and recommendation system using a fine-tuned ResNet50 for visual features and SentenceTransformers for text understanding.

## Prerequisites
- Python 3.9 or higher
- All dependencies installed in the virtual environment

## How to Start the System

### 1. Start the Backend Server
The backend is powered by FastAPI and handles the search logic using the fine-tuned models.
Run the following command in your terminal from the project root:
```bash
./venv/bin/python main.py
```
*The server will start at `http://localhost:8000`.*

### 2. Launch the Web Application
Once the server is running, you can open the premium search interface:
1. Navigate to the `ux` folder.
2. Open `index.html` in any modern web browser (Chrome, Safari, Edge).
   - Alternatively, use the absolute path: `/Users/dev_joshi/Desktop/project/fashion_recsys/ux/index.html`

## Features
- **Visual Search**: Drag and drop or upload an image to find visually similar items.
- **Natural Language Search**: Search for products using descriptive terms like "red dress" or "striped shirt".
- **Fine-tuned Precision**: Uses a model specifically trained on fashion categories for superior accuracy.

## Project Structure
- `main.py`: FastAPI server entry point.
- `ux/`: Modern frontend interface (HTML/CSS/JS).
- `data/`: Contains datasets and fine-tuned model weights.
- `search.py`: Core multimodal search logic.
- `model.py`: Model definitions and loading.