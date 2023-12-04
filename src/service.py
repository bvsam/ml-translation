import bentoml
import torch
from bentoml.io import JSON
from pydantic import BaseModel

model = bentoml.pytorch.get("en-fr_rnn_lstm_512:latest")
model_runner = model.to_runner()

svc = bentoml.Service("en-fr_translation", runners=[model_runner])


class Request(BaseModel):
    input: str


class Response(BaseModel):
    translation: str


@svc.api(input=JSON(pydantic_model=Request), output=JSON(pydantic_model=Response))
async def translate(request: Request) -> Response:
    prepare_input = model.custom_objects["prepare_input"]
    tensor_to_sentence = model.custom_objects["tensor_to_sentence"]
    eng_vocab = model.custom_objects["eng_vocab"]
    fra_vocab = model.custom_objects["fra_vocab"]

    input = torch.tensor(
        prepare_input(request.input, eng_vocab), dtype=torch.long
    ).unsqueeze(0)
    output = await model_runner.async_run(input)
    output = output.squeeze(0)
    _, pred_indexes = output.topk(1, dim=1)
    output = tensor_to_sentence(pred_indexes, fra_vocab)

    return Response(translation=output)
