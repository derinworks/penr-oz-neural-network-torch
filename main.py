from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.params import Query
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from asyncio import Lock, create_task
from typing import Dict
from neural_net_model import NeuralNetworkModel

app = FastAPI(
    title="Neural Network Model API",
    description="API to create, serialize, compute output and diagnose of neural network models.",
    version="0.0.1"
)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
logging.basicConfig(
    datefmt=DATE_FORMAT,
    format=LOG_FORMAT,
)
log = logging.getLogger(__name__)


# Constants for examples
EXAMPLES = [
    {
        "activation_vector": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "target_vector":     [0, 0, 0, 0, 1, 0, 0, 0, 0]
    },
    {
        "activation_vector": [0, 0, 0, 0,-1, 0, 0, 0, 0],
        "target_vector":     [1, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 0, 0, 0,-1, 0, 0, 0, 0],
        "target_vector":     [0, 1, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,-1, 0, 0,-1, 0, 0, 0, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 0, 1, 0]
    },
    {
        "activation_vector": [1,-1, 0, 0,-1, 0, 0, 1, 0],
        "target_vector":     [0, 0, 1, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,-1,-1, 0,-1, 0, 0, 1, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 1, 0, 0]
    },
    {
        "activation_vector": [1,-1,-1, 0,-1, 0, 1, 1, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 0, 0, 1]
    },
    {
        "activation_vector": [1,-1,-1, 0,-1, 0, 1, 1,-1],
        "target_vector":     [0, 0, 0, 1, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,-1,-1, 1,-1, 0, 1, 1,-1],
        "target_vector":     [0, 0, 0, 0, 0, 1, 0, 0, 0]
    },
]


class InputItem(BaseModel):
    activation_vector: list = Field(
        ...,
        description="The input activation data. 1D or 2D supported."
    )
    target_vector: list | None = Field(
        None,
        description="The optional expected target data same size as input data."
    )


class TrainingItem(InputItem):
    target_vector: list[float] = Field(
        ...,
        description="The expected output vector, required for training items."
    )


class ModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )


class CreateModelRequest(ModelRequest):
    layer_sizes: list[int] = Field(
        ...,
        examples=[[9, 1, 9, 9]],
        description="A list of integers representing the sizes of each layer in the neural network."
    )
    weight_algo: str = Field(
        "xavier",
        examples=["xavier", "he", "gaussian"],
        description="An initialization algorithm"
    )
    bias_algo: str = Field(
        "random",
        examples=["random", "zeros"],
        description="An initialization algorithm"
    )
    activation_algos: list[str] = Field(
        ...,
        examples=[["sigmoid"] * 3, ["relu"] * 2 + ["sigmoid"], ["relu"] * 2 + ["softmax"], ["embedding", "tanh"]],
        description="The activation algorithms to apply"
    )
    optimizer: str = Field(
        "adam",
        examples=["adam", "stochastic"],
        description="The optimizer to use for updating for gradient descent"
    )
    batchnorm_eps: float = Field(
        1e-5,
        examples=[1e-5],
        description="Batch Normalization Epsilon"
    )
    batchnorm_momentum: float = Field(
        0.1,
        examples=[0.1],
        description="Batch Normalization Momentum"
    )
    confidence: float = Field(
        1.0,
        examples=[1.0],
        description="Confidence factor for the last layer with weights"
    )


class ActivationRequest(ModelRequest):
    input: InputItem = Field(
        ...,
        description="The input data, an InputItem."
    )


class TrainingRequest(ModelRequest):
    training_data: list[TrainingItem] = Field(
        ...,
        examples=[EXAMPLES],
        description="A list of training data pairs."
    )
    epochs: int = Field(
        10,
        examples=[10],
        description="The number of training epochs."
    )
    learning_rate: float = Field(
        0.01,
        examples=[0.01],
        description="The learning rate for training."
    )
    batch_size: int | None = Field(
        None,
        examples=[32],
        description="The batch size for training sample each epoch. (Optional)"
    )
    decay_rate: float = Field(
        0.9,
        examples=[0.9],
        description="The decay rate of learning rate during training."
    )
    dropout_rate: float = Field(
        0.2,
        examples=[0.2],
        description="The drop out rate of activated neurons to improve generalization"
    )
    l2_lambda: float = Field(
        0.001,
        examples=[0.001],
        description="The L2 Lambda penalty reducing weight magnitude during backpropagation"
    )
    adam_beta1: float = Field(
        0.9,
        examples=[0.9],
        description="The Adam optimizer parameter for gradient mean optimization after backpropagation"
    )
    adam_beta2: float = Field(
        0.999,
        examples=[0.999],
        description="The Adam optimizer parameter for gradient variance optimization after backpropagation"
    )
    adam_epsilon: float = Field(
        1e-8,
        examples=[1e-8],
        description="The Adam optimizer parameter for smallest step for gradient optimization after backpropagation"
    )


class ModelIdQuery(Query):
    description="The unique identifier for the model."


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, e: Exception):
    log.error(f"An error occurred: {str(e)}")
    return JSONResponse(status_code=500, content={"detail": "Please refer to server logs"})


@app.exception_handler(KeyError)
async def key_error_handler(_: Request, e: KeyError):
    raise HTTPException(status_code=404, detail=f"Not found error occurred: {str(e)}")


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, e: ValueError):
    raise HTTPException(status_code=400, detail=f"Value error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
    return RedirectResponse(url="/dashboard")

# Dashboard route
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting creation of model {model_id}")
    model = NeuralNetworkModel(model_id, body.layer_sizes, body.weight_algo, body.bias_algo, body.activation_algos,
                               body.optimizer, (body.batchnorm_eps, body.batchnorm_momentum), body.confidence)
    model.serialize()
    return {"message": f"Model {model_id} created and saved successfully"}


@app.post("/output/")
def compute_model_output(body:
    ActivationRequest = Body(...,
                             openapi_examples={f"example_{idx}": {
                                 "summary": f"Example {idx + 1}",
                                 "description": f"Example input and training data for case {idx + 1}",
                                 "value": {
                                     "model_id": "test",
                                     "input": example
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    model_id = body.model_id
    log.info(f"Requesting output for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    input_vector = body.input.activation_vector
    target = body.input.target_vector
    output, cost = model.compute_output(input_vector, target)
    return {"output_vector": output,
            "cost": cost,
            }


# This will track active training sessions by model_id
model_locks: Dict[str, Lock] = {}

@app.put("/train/")
async def train_model(body: TrainingRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting training for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)

    # Get or create a lock for this model
    if model_id not in model_locks:
        model_locks[model_id] = Lock()
    lock = model_locks[model_id]

    # If the model is already locked (training), return 409 Conflict
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"Training already in progress for model {model_id}.")

    async def train():
        async with lock:
            await run_in_threadpool(
                model.train,
                [(data.activation_vector, data.target_vector) for data in body.training_data],
                body.epochs,
                body.learning_rate,
                body.batch_size,
                body.decay_rate,
                body.dropout_rate,
                body.l2_lambda,
                body.adam_beta1,
                body.adam_beta2,
                body.adam_epsilon,
            )

    # Start training in the background
    create_task(train())

    # Respond with request accepted
    return JSONResponse(content={"message": f"Training for model {model_id} started asynchronously."}, status_code=202)

@app.get("/progress/")
def model_progress(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting progress for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return {
        "progress": model.progress,
        "average_cost": model.avg_cost,
        "average_cost_history": model.avg_cost_history,
        "status": model.status,
    }

@app.get("/stats/")
def model_stats(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting stats for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return model.stats

@app.delete("/model/")
def delete_model(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting deletion of model {model_id}")
    NeuralNetworkModel.delete(model_id)
    return Response(status_code=204)


if __name__ == "__main__": # pragma: no cover
    import uvicorn

    uvicorn.run(app,
                host="127.0.0.1",
                port=8000,
                log_config={
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "default": {
                            "format": LOG_FORMAT,
                            "datefmt": DATE_FORMAT,
                        },
                    },
                    "handlers": {
                        "default": {
                            "level": "INFO",
                            "class": "logging.StreamHandler",
                            "formatter": "default",
                        },
                    },
                    "loggers": {
                        "uvicorn": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                        "uvicorn.error": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                        "uvicorn.access": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                    },
                    "root": {
                        "level": "INFO",
                        "handlers": ["default"],
                    },
                })
