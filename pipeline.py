from kfp import dsl
from kfp import compiler

@dsl.container_component
def preprocess_op(data_output: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image='your-docker-username/mnist-kfp:latest',
        command=['python', 'preprocess.py'],
        args=['--output_path', data_output.path]
    )

@dsl.container_component
def train_op(data_input: dsl.Input[dsl.Dataset], model_output: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        image='your-docker-username/mnist-kfp:latest',
        command=['python', 'train.py'],
        args=['--input_path', data_input.path, '--model_path', model_output.path]
    )

@dsl.container_component
def evaluate_op(data_input: dsl.Input[dsl.Dataset], model_input: dsl.Input[dsl.Model], metrics: dsl.Output[dsl.ClassificationMetrics]):
    return dsl.ContainerSpec(
        image='your-docker-username/mnist-kfp:latest',
        command=['python', 'evaluate.py'],
        args=['--model_path', model_input.path, '--data_path', data_input.path, '--metrics_path', metrics.path]
    )

@dsl.container_component
def deploy_op(model_input: dsl.Input[dsl.Model]):
    return dsl.ContainerSpec(
        image='your-docker-username/mnist-kfp:latest',
        command=['python', 'deploy.py'],
        args=['--model_path', model_input.path, '--deploy_path', '/tmp/prod_model']
    )

@dsl.pipeline(name='mnist-full-pipeline')
def mnist_pipeline():
    prep = preprocess_op()
    train = train_op(data_input=prep.outputs['data_output'])
    eval_task = evaluate_op(data_input=prep.outputs['data_output'], model_input=train.outputs['model_output'])
    deploy_task = deploy_op(model_input=train.outputs['model_output'])
    deploy_task.after(eval_task) # Only deploy if evaluation succeeds

if __name__ == '__main__':
    compiler.Compiler().compile(mnist_pipeline, 'pipeline.yaml')