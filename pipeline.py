from kfp import dsl
from kfp import compiler


@dsl.container_component
def preprocess_op(data_output: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image='mnist-pipeline:latest',
        command=['python', 'preprocess.py'],
        args=['--output_path', data_output.path]
    )


@dsl.container_component
def train_op(
    data_input: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model]
):
    return dsl.ContainerSpec(
        image='mnist-pipeline:latest',
        command=['python', 'train.py'],
        args=[
            '--input_path', data_input.path,
            '--model_path', model_output.path
        ]
    )


@dsl.container_component
def evaluate_op(
    data_input: dsl.Input[dsl.Dataset],
    model_input: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.ClassificationMetrics]
):
    return dsl.ContainerSpec(
        image='mnist-pipeline:latest',
        command=['python', 'evaluate.py'],
        args=[
            '--model_path', model_input.path,
            '--data_path', data_input.path,
            '--metrics_path', metrics.path
        ]
    )


@dsl.container_component
def deploy_op(model_input: dsl.Input[dsl.Model]):
    return dsl.ContainerSpec(
        image='mnist-pipeline:latest',
        command=['python', 'deploy.py'],
        args=[
            '--model_path', model_input.path,
            '--deploy_path', '/tmp/prod_model'
        ]
    )


@dsl.pipeline(name='mnist-pipeline')
def mnist_pipeline():

    prep_task = preprocess_op()

    train_task = train_op(
        data_input=prep_task.outputs['data_output']
    )

    eval_task = evaluate_op(
        data_input=prep_task.outputs['data_output'],
        model_input=train_task.outputs['model_output']
    )

    deploy_task = deploy_op(
        model_input=train_task.outputs['model_output']
    )

    # Disable caching (this is supported)
    for task in [prep_task, train_task, eval_task, deploy_task]:
        task.set_caching_options(False)

    # Ensure deployment runs after evaluation
    deploy_task.after(eval_task)


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path='pipeline.yaml'
    )
