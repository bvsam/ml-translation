# ML Translation

## About

- Sequence to sequence models (seq2seq) for translation
  - bidirectional encoder-decoder LSTM
    - notebook at `src/en-fr_rnn.ipynb`
- English to French translation using the Anki dataset
- Model deployment on AWS Sagemaker for inference using [bentoml](https://www.bentoml.com/)

## Deployment

Steps to deploy the trained model on AWS Sagemaker for inference. The final result will include the model being accessible for inference at a specific API endpoint on AWS.

### Prerequisites:

- Install Terraform
- Install Docker

### Steps

1. Install `requirements.txt`

```
pip install -r requirements.txt
```

2. Create the bentoml model

```
cd src
py create_model.py
```

3. Build the bento

```
bentoml build
```

4. Install the aws-sagemaker bentoctl operator

```
bentoctl operator install aws-sagemaker
```

5. Initialize a bentoctl deployment. You may want to delete files including `src/deployment_config.yaml` and `src/main.tf` before doing so (so that they can be overwritten without any conflict).

Note: If other operators are installed, running `bentoctl init` may not work as expected, specifically for Windows hosts. Remove all existing bentoctl operators by deleting the bentoctl folder at `~/bentoctl`. Then install the operator you'd like to work with.

```
bentoctl init
```

6. Run `bentoctl build` with the deployment config and the built bento. You can view existing bentos with the command `bentoml list`.

```
bentoctl build -f deployment_config.yaml -b <YOUR_BENTO_NAME:TAG_HERE>
```

7. Initialize Terraform and apply the terraform config/plan. Use `bentoctl.tfvars`, which should have been created when running `bentoctl init`, for the var. file.

```
terraform init
terraform apply --var-file=bentoctl.tfvars
```

### Teardown

Once done, destroy all resources created (including the AWS ecr repository) with:

```
bentoctl destroy
```

## TODO (Improvements)

- Implement attention with the RNNs, or even a transformer
- Try using a larger dataset, such as WMT 2014

## References

- Pytorch tutorial by [Sean Robertson](https://github.com/spro) introducing seq2seq networks: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- Youtube video by [@mildlyoverfitted](https://www.youtube.com/@mildlyoverfitted) explaining the bentoml deployment process: https://youtu.be/Zci_D4az9FU
