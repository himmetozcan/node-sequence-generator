import gradio as gr
import json
import random
from node_seq_gen import process_query, load_test_cases, run_tests, print_test_summary
from io import StringIO
import sys
import re
import unittest
from test_module import TestNodeSeqSystem

try:
    with open('test_prompts.json', 'r') as f:
        ALL_TEST_CASES = json.load(f)
except:
    ALL_TEST_CASES = [
        {"User Prompt": "When a button is clicked, fetch data and display it in a modal"},
        {"User Prompt": "Filter a list of items and show the results"},
        {"User Prompt": "Play a sound when the user presses a key"}
    ]

def load_examples():
    return [case["User Prompt"] for case in random.sample(ALL_TEST_CASES, 5)]

def generate_sequence(prompt, max_attempts, validation_threshold, show_steps=False):
    if not prompt.strip():
        return "Please enter a prompt."
    
    model = "qwen2.5-coder:7b"
    node_sequence, was_validated, debug_output = process_query(
        prompt,
        max_attempts=max_attempts,
        validation_threshold=validation_threshold,
        selected_model=model,
        silent=not show_steps
    )
    
    if node_sequence is None:
        return "❌ Failed to generate sequence."
    
    validation_status = "✅ Validated" if was_validated else "⚠️ Not fully validated"
    result = f"{validation_status}\n\n{chr(10).join(node_sequence)}"
    
    if show_steps:
        result = f"{debug_output}\n\n{'-' * 40}\n\n{result}"
    
    return result

def run_unit_tests():
    # Capture stdout to get the test results
    old_stdout = sys.stdout
    result_output = StringIO()
    sys.stdout = result_output
    
    # Run the unittest suite with verbosity=2 for detailed output
    suite = unittest.TestLoader().loadTestsFromName('test_module.TestNodeSeqSystem')
    unittest.TextTestRunner(stream=result_output, verbosity=2).run(suite)
    
    # Restore stdout and get output
    sys.stdout = old_stdout
    test_output = result_output.getvalue()
    
    # Strip ANSI color codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', test_output)
    
    # Format the output for display
    if not clean_output:
        return "No test results available."
    
    return clean_output

with gr.Blocks(css="""
    #generate-btn {
        border-radius: 50%; 
        min-width: 60px; 
        max-width: 60px; 
        height: 60px;
        margin-top: 60px;
        padding: 0;
        font-size: 24px;
    }
    #prompt-row {gap: 5px;}
    .container {
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
""") as iface:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Node Sequence Generator")

        output_text = gr.Textbox(
            lines=4,
            label="Generated Sequence"
        )

        with gr.Row(elem_id="prompt-row"):
            input_text = gr.Textbox(
                lines=3,
                placeholder="Enter your prompt here... (Press Enter to generate)",
                label="Prompt",
                scale=20,
                interactive=True
            )
            submit_btn = gr.Button("▶", elem_id="generate-btn", scale=1)

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                max_attempts_input = gr.Number(
                    value=10,
                    label="Max Attempts",
                    minimum=1,
                    maximum=20,
                    step=1
                )
                validation_threshold_input = gr.Number(
                    value=5,
                    label="Validation Threshold",
                    minimum=1,
                    maximum=10,
                    step=1
                )
                show_steps_checkbox = gr.Checkbox(
                    label="Show Steps",
                    value=False
                )
                run_tests_btn = gr.Button("Run Unit Tests")

        examples_state = gr.State(load_examples())
        examples_radio = gr.Radio(
            choices=examples_state.value,
            label="Examples (select one)",
            interactive=True
        )

        refresh_btn = gr.Button("Refresh Examples")

        # Event handlers
        examples_radio.change(fn=lambda x: x, inputs=examples_radio, outputs=input_text)
        submit_btn.click(
            fn=generate_sequence,
            inputs=[input_text, max_attempts_input, validation_threshold_input, show_steps_checkbox],
            outputs=output_text
        )
        input_text.submit(
            fn=generate_sequence,
            inputs=[input_text, max_attempts_input, validation_threshold_input, show_steps_checkbox],
            outputs=output_text
        )
        refresh_btn.click(
            fn=lambda: gr.update(choices=load_examples(), value=None),
            inputs=None,
            outputs=examples_radio
        )
        run_tests_btn.click(
            fn=run_unit_tests,
            inputs=None,
            outputs=output_text
        )

iface.launch(
        server_name="0.0.0.0",  # Very important - allows external connections
        server_port=7860,
        share=True  # Set to True if you want a public URL
    )
