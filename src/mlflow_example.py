import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_mlflow():
    try:
        # Get MLflow configuration
        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        
        # Set up the tracking URI to use the MLflow tracking server
        tracking_uri = f"http://localhost:{mlflow_port}"
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set the experiment name
        experiment_name = "solar_panel_detection"
        logger.info(f"Setting experiment name to: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        # Test the connection
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            logger.info(f"Found existing experiment with ID: {experiment.experiment_id}")
        else:
            logger.info("Creating new experiment")
        logger.info("Successfully connected to MLflow tracking server")
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def train_and_log_model():
    try:
        # Generate simple sample data
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Start an MLflow run
        with mlflow.start_run(run_name="simple_regression") as run:
            logger.info(f"Started MLflow run with ID: {run.info.run_id}")
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Log parameters
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("random_state", 42)
            logger.info("Logged parameters")

            # Log metrics
            mlflow.log_metric("mse", mse)
            logger.info(f"Logged metric MSE: {mse}")

            # Log the model
            mlflow.sklearn.log_model(model, "model")
            logger.info("Logged model")

    except Exception as e:
        logger.error(f"Error in train_and_log_model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set up MLflow with PostgreSQL
        setup_mlflow()
        
        # Train and log the model
        train_and_log_model()
        
        logger.info("Model training completed and logged to MLflow!")
        logger.info("You can view the results by running: mlflow ui")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 