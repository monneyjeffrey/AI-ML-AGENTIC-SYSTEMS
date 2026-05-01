# Estimated Time Of Arrival(ETA) Predictor

ETA Predictor is a FastAPI-based machine learning(ML) service that predicts the estimated delivery time (ETA) for logistics in Ghana.

ETA Predictor shows the pickup and delivery location, time of departure, vehicle type, etc.
It will then show the estimated time of arrival in minutes


eta-predictor/
├── app/
│   ├── main.py        # FastAPI application (API endpoints)
│   ├── schemas.py     # Request/response validation models
│   └── predictor.py  # ML model loading & prediction
├── scripts/
│   ├── generate_data.py  # Create synthetic training data
│   ├── preprocess.py    # Prepare data for training
│   └── train.py        # Train ML model
├── tests/
│   ├── test_api.py      # API endpoint tests
│   └── test_schemas.py # Validation tests
├── data/               # Raw and processed data
├── models/             # Trained model files
├── dvc.yaml           # DVC pipeline definition
├── Dockerfile         # Container image definition
└── docker-compose.yml # Local development stack



http://localhost:8000/
