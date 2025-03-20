# Flask-RESTful API Example

This is a simple Flask-RESTful API for managing a to-do list. The API supports creating, reading, updating, and deleting to-do items.

## Prerequisites

Before you begin, make sure you have the following installed:

- [Conda PKG for Mac Metal](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)
- Python 3.7 or higher

## Getting Started

Follow these steps to set up the project and run the Flask-RESTful API.

### Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/mrasyadc/OculabImageStitchingPredictionBackend

cd OculabImageStitchingPredictionBackend
```

### Step 2: Create a Conda Environment and Install Dependencies

Create a Conda Environment:
To create a new Conda environment with Python 3.9, run the following command:

```bash
conda create --name flask_restful
```

Activate the Environment:
After creating the environment, activate it:

```bash
conda activate flask_restful
```

Install Flask:
Once the environment is activated, install Flask using Conda:

```bash
conda install flask
```

Install requirements.txt:
Since Flask-RESTful, Ultralytics model for YOLO is not available in the default Conda packages, install it using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Run the API
After setting up the environment and installing the dependencies, you can now run the Flask-RESTful API using the example.py file.

```bash
conda activate flask_restful
python routes.py
```

The API will be running locally at http://127.0.0.1:5000.

### Step 4: Example Usage
You can interact with the API using curl or Postman.

GET all to-do items:

```bash
curl http://127.0.0.1:5000/todos
```
POST a new to-do item:
```bash
curl -X POST http://127.0.0.1:5000/todos -d "task=Learn Flask"
```
GET a specific to-do item:
```bash
curl http://127.0.0.1:5000/todos/todo1

```
PUT (update) a to-do item:
```bash
curl -X PUT http://127.0.0.1:5000/todos/todo1 -d "task=Finish API"
```
DELETE a to-do item:
```bash

curl -X DELETE http://127.0.0.1:5000/todos/todo1
```
Project Structure
```bash
flask-restful-api/
│
├── example.py      # Example API file
├── model.py        # Model Prediction (YOLO) API file
├── routes.py       # Main Entry Routes API file
├── README.md       # Instructions for setup
└── .gitignore      # Optional, includes .DS_Store, env, etc.
```

License
This project is licensed under the MIT License.


This README provides all the steps necessary to create a Conda environment, install Flask and Flask-RESTful, and run your API. Let Rasyad know if you need any further assistance!





