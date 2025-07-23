import gradio as gr
import os

# Set custom temp directory for Gradio caching
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

def compute_cost(default_training_time, generations, population_size, dataset_ratio, epochs_ratio, speedup_ratio, price_per_hour):
    try:
        # Calculate training time
        training_time = (
            default_training_time
            * generations
            * population_size
            * dataset_ratio
            * epochs_ratio
            / speedup_ratio
        )

        # Calculate cost
        result = training_time * price_per_hour
        result = round(result, 2)
        training_time = round(training_time, 2)

        # Choose image based on cost
        image = "stonks.png" if result < 1000 else "not_stonks.png"
        return result, training_time, image

    except ZeroDivisionError:
        return "Error: Speedup ratio cannot be zero.", None, None

with gr.Blocks() as demo:
    gr.Markdown("# 🧮 GA Training Cost Estimator")

    with gr.Row():
        default_training_time = gr.Number(label="Default Training Time (hours)", value=36)
        generations = gr.Number(label="Generations", value=300)
        population_size = gr.Number(label="Population Size", value=50)

    with gr.Row():
        dataset_ratio = gr.Number(label="Dataset Ratio", value=0.1)
        epochs_ratio = gr.Number(label="Epochs Ratio", value=0.2)
        speedup_ratio = gr.Number(label="Speedup Ratio", value=10)

    price_per_hour = gr.Number(label="Price per Hour ($)", value=2)

    with gr.Row():
        output = gr.Number(label="Estimated Total Cost ($)", interactive=False)
        training_output = gr.Number(label="Estimated Training Time (hrs)", interactive=False)
        image_output = gr.Image(label="Visualization", type="filepath")

    calc_button = gr.Button("Calculate")

    calc_button.click(
        fn=compute_cost,
        inputs=[
            default_training_time, generations, population_size,
            dataset_ratio, epochs_ratio, speedup_ratio, price_per_hour
        ],
        outputs=[output, training_output, image_output]
    )

demo.launch(server_name="0.0.0.0")
