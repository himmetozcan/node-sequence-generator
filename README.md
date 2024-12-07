# Node Sequence Generator

Demo for translating natural language instructions into structured sequences of predefined nodes.

## Features

- **AI-Powered**: Using qwen2.5-coder:7b model through ollama.
- **Validation**: Ensures generated sequences meet predefined criteria.
- **Interactive Web Interface**: Built with [Gradio](https://gradio.app) for user-friendly interaction.
- **Terminal Mode**: Command-line interface for development and testing.
- **Testing Suite**: Automated testing for robust performance.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/himmetozcan/node-sequence-generator.git
   cd node-sequence-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install and start Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   sudo systemctl start ollama

   # Pull the required AI model
   ollama pull qwen2.5-coder:7b
   ```

### Model Specifications
- **Model Size**: 7.6B parameters
- **Quantization**: Q4_K_M (4-bit quantization with medium accuracy)
- **VRAM Requirements**: ~6GB VRAM
- **Context Length**: 32,768 tokens
- **Embedding Length**: 3,584

4. Choose your preferred mode:

   **A. Gradio Web Interface**:
   ```bash
   # Build the Docker image
   sudo docker build -t node-seq-gen-app .

   # Run the container
   sudo docker run -it --rm \
       --network host \
       -e OLLAMA_HOST=http://localhost:11434 \
       node-seq-gen-app
   ```
   Then open [localhost:7860](http://localhost:7860) in your browser.

   **B. Interactive Terminal Mode**:
   ```bash
   python node_seq_gen.py
   ```

## Usage

### Operation Modes

1. **Gradio Web Interface**:
   - User-friendly web interface for sequence generation
   - Access through your browser at [localhost:7860](http://localhost:7860)
   - Simple input field for task descriptions
   - Visual display of generated sequences

2. **Interactive Terminal Mode**:
   - Direct command-line interface
   - Features:
     - Switch between different LLM models
     - Interactive prompt input
     - Run test suites
     - Debug output and detailed logging
   - Start with: `python node_seq_gen.py`

### Basic Operation
- **Input**: Describe a task (e.g., "Fetch data and display it in a modal").
- **Output**: The system generates a sequence of nodes (e.g., `["FetchData", "DisplayModal"]`).

## Testing

Run the tests:
```bash
python -m unittest test_module.py
```

### Performance Analysis

The test suite uses `test_prompts.json`, which contains 100 test prompts generated using GPT-4o and manually reviewed. Note that these test prompts are distinct from the examples in the system message to better evaluate generalization.

Performance metrics with different configurations:

1. **Balanced Configuration** (Default)
   - Max Attempts: 10
   - Validation Threshold: 5
   - Mean Accuracy: 69.0%
   - Standard Deviation: 1.58%

2. **Medium Configuration**
   - Max Attempts: 7
   - Validation Threshold: 4
   - Mean Accuracy: 60.4%
   - Standard Deviation: 2.88%

3. **Single-Pass Configuration**
   - Max Attempts: 1
   - Validation Threshold: 0
   - Mean Accuracy: 61.8%
   - Standard Deviation: 4.09%

The results demonstrate that increasing max attempts and validation threshold:
- Improves overall accuracy
- Reduces output variability (lower standard deviation)
- Provides more stable and reliable results
- More information about the algorithm is given below.

### Algorithm Overview

This script uses a Large Language Model (LLM), accessed via the [Ollama](https://ollama.ai) client, to convert a user’s textual request into a sequence of predefined “nodes.” These nodes represent actions, events, transformations, and display operations within a hypothetical application. The idea is to have the LLM infer the correct sequence of these nodes from natural language instructions, verify the correctness, and then present the final node sequence to the user.

### Key Points

1. **Ollama Integration**:  
   The script uses the `ollama` Python client to communicate with an Ollama server (by default at `http://localhost:11434`). Through this, it sends messages to the LLM and receives its responses.

2. **System Message and Role-based Prompts**:  
   A large and carefully crafted system message is defined at the start. This message provides the LLM with detailed instructions:
   - It explains the concept of nodes and their categories (Event, Action, Transformation, Display, Data).
   - It specifies how the LLM must respond: always output a JSON object or array with a “sequence” of node names, never adding extraneous text.
   - It provides numerous example user prompts with the correct expected output sequences.

   By defining the system message, the script ensures that the LLM understands the allowed node types and the required output format.

3. **User Query Processing**:
   The user provides a prompt describing what they want to achieve (e.g., "Fetch data and show it on the screen"). The script then:
   - Sends this prompt to the LLM, along with the system instructions.
   - The LLM’s job is to return a JSON sequence of nodes that accomplish the user’s request.

4. **Response Parsing and Validation**:
   Once the LLM responds, the script:
   - Attempts to parse the LLM response as JSON.
   - Extracts the “sequence” of nodes from the JSON.
   - Validates that all nodes exist in the known set of allowed nodes.
   
   After parsing, the script runs an additional validation step:
   - It sends the generated node sequence back to the LLM with a request to validate correctness.
   - The LLM returns a JSON object indicating whether the sequence is valid (`{"valid": true}` or `{"valid": false}`).
   
   This creates a feedback loop where the script can try multiple times (by default, up to 3 attempts) to get a valid sequence if the initial attempts fail.

5. **Retry and Consensus Logic**:
   The script has a built-in retry mechanism:
   - It tries up to `max_attempts` times to get a valid sequence.
   - Each attempt involves generating a sequence and validating it multiple times (`validation_threshold`).
   
   If no fully validated sequence can be achieved after the maximum number of attempts, the script looks at all generated sequences and picks the most commonly generated one as a fallback result. This ensures that the user at least gets the most frequent “best guess,” even if it didn’t pass full validation.

6. **Running Test Cases**:
   The script can load test prompts from a `test_prompts.json` file. Each test prompt has an expected correct node sequence. Using multiprocessing, it can:
   - Process each test prompt, attempting to generate and validate a node sequence.
   - Compare the generated sequence to the expected sequence.
   
   Results are tallied, printed to the console, and saved to a CSV file. This test mode helps ensure that the LLM model and prompt configuration produce consistent and correct outputs over a range of predefined scenarios.

7. **Interactive Mode**:
   When run interactively (simply running `python node_seq_gen.py`):
   - The user can switch between different predefined models.
   - The user can type in requests and see the generated node sequences.
   - The user can run the test suite to see how well the LLM performs on known test cases.


### Typical Workflow

1. **User Input**: The user enters a prompt, such as:  
   "Fetch user data and display it in a modal."
   
2. **LLM Generation**:  
   The script sends the request and instructions to the LLM. The LLM returns something like:  
   `{"sequence": ["OnClick", "FetchData", "DisplayModal"]}`
   
3. **Validation**:  
   The script checks that all nodes (`OnClick`, `FetchData`, `DisplayModal`) are allowed. Then it asks the LLM if this sequence fulfills the user request. If `valid: true` returns, great—done. Otherwise, it retries.

4. **Final Output**:  
   The sequence is printed out for the user, and if it passed validation, the user is confident it matches the intended functionality.





### Available Nodes

#### Event Nodes
- `OnVariableChange`: Triggers on variable value changes
- `OnKeyPress/Release`: Keyboard event handlers
- `OnClick`: Click event handler
- `OnWindowResize`: Window resize event handler
- `OnMouseEnter/Leave`: Mouse hover event handlers
- `OnTimer`: Time-based triggers
- `Delay`: Execution delay

#### Action Nodes
- `Console/Alert/Log`: Output and debugging
- `Assign`: Variable assignment
- `SendRequest`: Network requests
- `Navigate`: Page navigation
- `Save/Delete`: Data persistence
- `PlaySound/PauseSound/StopSound`: Audio control

#### Transformation Nodes
- `Branch`: Conditional logic
- `Map/Filter/Reduce`: Data transformation
- `Sort/GroupBy`: Data organization
- `Merge/Split`: Data combination/separation

#### Display Nodes
- `Show/Hide/Update`: Display control
- `DisplayModal/CloseModal`: Modal dialogs
- `Highlight`: Element highlighting
- `Tooltip`: Additional information
- `RenderChart`: Data visualization

#### Data Nodes
- `FetchData`: Data retrieval
- `StoreData`: Data storage
- `UpdateData`: Data modification
- `DeleteData`: Data removal
- `CacheData`: Performance optimization

## License

This project is licensed under [MIT License](LICENSE).

