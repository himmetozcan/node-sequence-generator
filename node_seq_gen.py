import ollama
from rich.console import Console
from rich.markdown import Markdown
import pandas as pd
import json
from collections import Counter
from multiprocessing import Pool, cpu_count
import os

# Get Ollama host from environment variable or use default
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=ollama_host)
console = Console()


# System message for the LLM
system_message = """
You are a system that converts a user's instruction into a sequence of nodes.
Use a step by step approach. First define the actions in an order that achieves the user's goal. 
Then assign the nodes to the actions.
In final step, you must respond with a JSON object containing only a "sequence" key with an array of node names.
You MUST always output exactly and only a JSON array where each element is a node string. 
No explanations, no introductions, no trailing text.

Example Use Cases:

User Prompt: "Navigate to a new page after a delay of 3 seconds when the user clicks a button."
Correct Output: ["OnClick","Delay","Navigate"]

User Prompt: "Fetch user data and display it in a modal when a button is clicked."
Correct Output: ["OnClick","FetchData","DisplayModal"]

User Prompt: "Reduce a list of scores to find the highest score and log the result."  
Correct Output: ["Reduce", "Log"]

User Prompt: "Cache fetched data to improve performance and display the data on the screen."  
Correct Output: ["FetchData", "CacheData", "Show"]

User Prompt: "Log a message when a key is pressed and display the key value on the screen."  
Correct Output: ["OnKeyPress", "Log", "Show"]

User Prompt: "Highlight an element when the mouse enters it and remove the highlight when the mouse leaves."  
Correct Output: ["OnMouseEnter", "Highlight", "OnMouseLeave", "Show"]

User Prompt: "Filter out items that are out of stock and sort the remaining items by price before displaying them on the screen"
Correct Output: ["Filter","Sort","Show"]

You are only allowed to use the following nodes:

Event Nodes:
  [OnVariableChange]: Triggered when a specified variable changes value. 
  [OnKeyRelease]: Triggered when a key is released. 
  [OnKeyPress]: Triggered when a key is pressed. 
  [OnClick]: Triggered when an element is clicked. 
  [OnWindowResize]: Triggered when the window is resized. 
  [OnMouseEnter]: Triggered when the mouse pointer enters an element. 
  [OnMouseLeave]: Triggered when the mouse pointer leaves an element. 
  [OnTimer]: Triggered at specified time intervals.
  [Delay]: Delays the execution of the next node by a specified amount of time.

Action Nodes:
  [Console]: Prints a message to the console. 
  [Alert]: Displays an alert message. 
  [Log]: Logs information for debugging purposes. 
  [Assign]: Assigns a value to a variable. 
  [SendRequest]: Sends a network request. 
  [Navigate]: Navigates to a different URL or page. 
  [Save]: Saves data to local storage or a database. 
  [Delete]: Deletes specified data or records. 
  [PlaySound]: Plays an audio file. 
  [PauseSound]: Pauses an audio file. 
  [StopSound]: Stops an audio file.

Transformation Nodes: 
  [Branch]: Conditional node that branches based on a true/false evaluation. 
  [Map]: Transforms data from one format to another. 
  [Filter]: Filters data based on specified criteria. 
  [Reduce]: Reduces a list of items to a single value. 
  [Sort]: Sorts data based on specified criteria. 
  [GroupBy]: Groups data by a specified attribute. 
  [Merge]: Merges multiple datasets into one. 
  [Split]: Splits data into multiple parts based on criteria.

Display Nodes: 
  [Show]: Displays information on the screen. 
  [Hide]: Hides information from the screen. 
  [Update]: Updates the display with new information. 
  [DisplayModal]: Displays a modal dialog. 
  [CloseModal]: Closes an open modal dialog. 
  [Highlight]: Highlights an element on the screen. 
  [Tooltip]: Shows a tooltip with additional information. 
  [RenderChart]: Renders a chart with specified data.

Data Nodes: 
  [FetchData]: Fetches data from an API or database. 
  [StoreData]: Stores data in a variable or storage. 
  [UpdateData]: Updates existing data. 
  [DeleteData]: Deletes specified data. 
  [CacheData]: Caches data for performance improvement.


More Examples:
Below are 30 user prompts, each followed by the correct sequence of nodes. They range from simple scenarios to more complex ones, mixing event nodes with action, transformation, display, and data nodes.

User Prompt: "Fetch data from the server and then display it on the screen."  
Correct Output: ["FetchData","Show"]

User Prompt: "Send a network request, map the response, and then render a chart with the transformed data."  
Correct Output: ["SendRequest","Map","RenderChart"]

User Prompt: "Fetch data, filter out inactive items, and then show the filtered results."  
Correct Output: ["FetchData","Filter","Show"]

User Prompt: "Fetch data, reduce it to a summary count, and then display the result in a modal."  
Correct Output: ["FetchData","Reduce","DisplayModal"]

User Prompt: "Fetch data, sort it by name, and then highlight the top item."  
Correct Output: ["FetchData","Sort","Highlight"]

User Prompt: "Fetch data, branch logic to check if count > 10, if true display a modal, otherwise show a tooltip."  
Correct Output: ["FetchData","Branch","DisplayModal","Tooltip"]

User Prompt: "Fetch data, group it by category, then render a chart of counts per category."  
Correct Output: ["FetchData","GroupBy","RenderChart"]

User Prompt: "Send a request, merge its response with local data, and then show the merged result."  
Correct Output: ["SendRequest","Merge","Show"]

User Prompt: "Fetch data, map it to a simpler structure, then display it."  
Correct Output: ["FetchData","Map","Show"]

User Prompt: "Send a request, cache the response, and then update the display with the cached data."  
Correct Output: ["SendRequest","CacheData","Update"]

User Prompt: "Fetch data, split it into two arrays based on a condition, and then show the first array."  
Correct Output: ["FetchData","Split","Show"]

User Prompt: "Assign a value to a variable, then log that value."  
Correct Output: ["Assign","Log"]

User Prompt: "Fetch data, filter for premium users only, then display them in a modal."  
Correct Output: ["FetchData","Filter","DisplayModal"]

User Prompt: "Store data from a local source, map the stored data, and show the mapped data."  
Correct Output: ["StoreData","Map","Show"]

User Prompt: "Fetch data, branch on a condition: if true display a chart, else show an alert."  
Correct Output: ["FetchData","Branch","RenderChart","Alert"]

User Prompt: "Play a sound, after some processing pause it, then show a message that playback ended."  
Correct Output: ["PlaySound","PauseSound","Show"]

User Prompt: "Send a request, reduce the returned items to a single total, and log the total."  
Correct Output: ["SendRequest","Reduce","Log"]

User Prompt: "Fetch data, sort by price, display it, then show a tooltip for additional info."  
Correct Output: ["FetchData","Sort","Show","Tooltip"]

User Prompt: "Branch logic based on a stored variable, if true highlight an element, otherwise hide it."  
Correct Output: ["Branch","Highlight","Hide"]

User Prompt: "Fetch data, merge it with another dataset, then display the merged set."  
Correct Output: ["FetchData","Merge","Show"]

User Prompt: "Send a request, assign part of the response to a variable, and then update the display with it."  
Correct Output: ["SendRequest","Assign","Update"]

User Prompt: "Fetch data, filter out outdated entries, sort the remainder, and show the sorted list."  
Correct Output: ["FetchData","Filter","Sort","Show"]

User Prompt: "Cache some data, then render a chart of that cached data."  
Correct Output: ["CacheData","RenderChart"]

User Prompt: "Send a request, map the response fields, and then display a modal with the mapped data."  
Correct Output: ["SendRequest","Map","DisplayModal"]

User Prompt: "Fetch data, reduce it to a single metric, highlight an element representing that metric, and then show it."  
Correct Output: ["FetchData","Reduce","Highlight","Show"]

User Prompt: "Fetch data, split it into multiple parts, sort one part, and update the display with that sorted part."  
Correct Output: ["FetchData","Split","Sort","Update"]

User Prompt: "Store data locally, filter it for 'active' items, then show those items."  
Correct Output: ["StoreData","Filter","Show"]

User Prompt: "Fetch data, branch to decide if we should render a chart or display a modal, then execute the chosen action."  
Correct Output: ["FetchData","Branch","RenderChart","DisplayModal"]

User Prompt: "Send a request, group the response items, and show them grouped by category."  
Correct Output: ["SendRequest","GroupBy","Show"]

User Prompt: "Fetch data, update it with new information, and then display the updated data."  
Correct Output: ["FetchData","UpdateData","Show"]

User Prompt: "Send a request, map the result, cache it, and then render a chart of the cached data."  
Correct Output: ["SendRequest","Map","CacheData","RenderChart"]

"""

def get_llm_response(user_query, selected_model):
    """Get node sequence from LLM"""
    
    user_message = f"""
        User request: {user_query}

        Convert this request into a sequence of nodes that achieves the desired functionality.
        You are only allowed to use the nodes defined in the system message. They are:
        
        {get_available_nodes()}
        
        You must respond with a JSON object containing only a "sequence" key with an array of node names.
        
        Example format:
        {{
            "sequence": ["OnClick", "FetchData", "DisplayModal"]
        }}

        Use only the node names defined in the system message. Ensure the sequence is logical and achieves the user's goal.
    """

    response = client.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    return response['message']['content'].strip()

def parse_llm_response(content):
    """Parse LLM response to extract node sequence"""
    try:
        # Clean up markdown formatting if present
        if content.startswith('```json'):
            content = content.replace('```json', '', 1)
            content = content.replace('```', '', 1)
        elif content.startswith('```python'):
            content = content.replace('```python', '', 1)
            content = content.replace('```', '', 1)
        elif content.startswith('```'):
            content = content.replace('```', '', 1)
            content = content.replace('```', '', 1)
            
        content = content.strip()
        
        # First try parsing as JSON
        try:
            json_data = json.loads(content)
            if isinstance(json_data, list):
                return json_data
            elif isinstance(json_data, dict) and 'sequence' in json_data:
                return json_data['sequence']
        except json.JSONDecodeError:
            # If JSON parsing fails, try evaluating as Python list
            if content.startswith('[') and content.endswith(']'):
                # Using ast.literal_eval is safer than eval()
                import ast
                return ast.literal_eval(content)
            
        raise ValueError("Invalid response format")
    except Exception as e:
        raise ValueError(f"Failed to parse response: {str(e)}")

def validate_node_sequence(user_query, node_sequence, selected_model):
    """Validate if the node sequence is appropriate for the user query"""
    
    user_message = f"""
        Original User Request: "{user_query}"
        Generated Node Sequence: {node_sequence}

        Evaluate if this sequence correctly fulfills the user's request.
        Return ONLY a JSON object with a single "valid" boolean key.
        Example: {{"valid": true}}
    """

    try:
        response = client.chat(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Parse the response
        content = response['message']['content'].strip()
        if content.startswith('```json'):
            content = content.replace('```json', '', 1)
            content = content.replace('```', '', 1)
        
        validation_result = json.loads(content)
        return validation_result.get('valid', False)
        
    except Exception as e:
        return False

def get_available_nodes():
    """Returns a set of all available node names"""
    return {
        # Event Nodes
        "OnVariableChange", "OnKeyRelease", "OnKeyPress", "OnClick", 
        "OnWindowResize", "OnMouseEnter", "OnMouseLeave", "OnTimer", "Delay",
        
        # Action Nodes
        "Console", "Alert", "Log", "Assign", "SendRequest", "Navigate",
        "Save", "Delete", "PlaySound", "PauseSound", "StopSound",
        
        # Transformation Nodes
        "Branch", "Map", "Filter", "Reduce", "Sort", "GroupBy",
        "Merge", "Split",
        
        # Display Nodes
        "Show", "Hide", "Update", "DisplayModal", "CloseModal",
        "Highlight", "Tooltip", "RenderChart",
        
        # Data Nodes
        "FetchData", "StoreData", "UpdateData", "DeleteData", "CacheData"
    }

def validate_node_names(node_sequence):
    """Validate if all nodes in the sequence are available nodes"""
    available_nodes = get_available_nodes()
    invalid_nodes = []
    
    for node in node_sequence:
        if node not in available_nodes:
            invalid_nodes.append(node)
    
    return len(invalid_nodes) == 0, invalid_nodes

def process_query(user_query, max_attempts=3, validation_threshold=3, selected_model=None, silent=False):
    """Process user query with retry logic and multiple validations"""
    attempt = 0
    all_sequences = []
    debug_output = []
    
    def add_debug(message, type='info'):
        if not silent:
            if type == 'success':
                marker = '✅'
                color = 'green'
            elif type == 'error':
                marker = '❌'
                color = 'red'
            elif type == 'warning':
                marker = '⚠️'
                color = 'yellow'
            else:
                marker = '📝'
                color = 'blue'
            
            debug_output.append(f"{marker} {message}")
    
    while attempt < max_attempts:
        attempt += 1
        add_debug(f"Attempt {attempt}/{max_attempts}", 'info')
            
        try:
            llm_response = get_llm_response(user_query, selected_model)
            node_sequence = parse_llm_response(llm_response)
            
            nodes_valid, invalid_nodes = validate_node_names(node_sequence)
            if not nodes_valid:
                add_debug("Invalid nodes in sequence:", 'error')
                add_debug(f"Unknown nodes: {invalid_nodes}", 'error')
                continue
            
            all_sequences.append(tuple(node_sequence))
            
            add_debug("Generated Node Sequence:", 'info')
            add_debug(f"{str(node_sequence)}", 'info')
            
            validations_passed = 0
            for validation_num in range(1, validation_threshold + 1):
                add_debug(f"Validation {validation_num}/{validation_threshold}...", 'info')
                is_valid = validate_node_sequence(user_query, node_sequence, selected_model)
                
                if is_valid:
                    validations_passed += 1
                    add_debug(f"Validation {validation_num} successful!", 'success')
                else:
                    add_debug(f"Validation {validation_num} failed", 'error')
                    break
            
            if validations_passed == validation_threshold:
                add_debug(f"All {validation_threshold} validations passed!", 'success')
                return node_sequence, True, "\n".join(debug_output)
            elif attempt < max_attempts:
                add_debug("Retrying with new sequence...", 'warning')
                
        except Exception as e:
            add_debug(f"Error in attempt {attempt}: {str(e)}", 'error')
    
    add_debug(f"Could not achieve {validation_threshold} validations in {max_attempts} attempts", 'warning')
    add_debug("Analyzing most frequent sequence...", 'info')
    
    if all_sequences:
        sequence_counts = Counter(all_sequences)
        most_common_sequence = list(sequence_counts.most_common(1)[0][0])
        
        add_debug("Sequence Statistics:", 'info')
        for seq, count in sequence_counts.items():
            add_debug(f"Sequence {list(seq)}: {count} occurrences", 'info')
        
        return most_common_sequence, False, "\n".join(debug_output)
    
    return None, False, "\n".join(debug_output)

def load_test_cases(json_file='test_prompts.json', max_tests = 100):
    """Load test cases from JSON file"""
    if max_tests > 100:
        max_tests = 100
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        if max_tests > 0:
            test_cases = test_cases[:max_tests]
        return test_cases
    except Exception as e:
        console.print(f"[bold red]Error loading test cases: {str(e)}[/bold red]")
        return None

def process_single_test(args):
    """Helper function to process a single test case"""
    test_case, selected_model = args
    try:
        node_sequence, was_validated, debug_output = process_query(
            test_case['User Prompt'],
            max_attempts=3,
            validation_threshold=3,
            selected_model=selected_model
        )
        
        expected_sequence = test_case['Correct Output']
        passed = node_sequence == expected_sequence
        
        return {
            'prompt': test_case['User Prompt'],
            'expected': expected_sequence,
            'actual': node_sequence,
            'passed': passed,
            'validated': was_validated,
            'error': None
        }
    except Exception as e:
        return {
            'prompt': test_case['User Prompt'],
            'error': str(e)
        }

def run_tests(test_cases, selected_model=None):
    """Run tests in parallel and return performance metrics"""
    if not test_cases:
        return None
    
    results = {
        'total_tests': len(test_cases),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': []
    }

    # Create arguments for each test case
    test_args = [(test_case, selected_model) for test_case in test_cases]
    
    # Use number of CPU cores for parallel processing
    num_processes = cpu_count()
    console.print(f"[bold blue]Running tests using {num_processes} processes...[/bold blue]")
    
    # Run tests in parallel
    with Pool(processes=num_processes) as pool:
        test_results = list(pool.imap_unordered(process_single_test, test_args))
    
    # Process results
    for result in test_results:
        if 'error' in result and result['error'] is not None:
            results['errors'] += 1
            console.print(f"[bold red]Error in test[/bold red]")
        else:
            if result['passed']:
                results['passed'] += 1
                status = "[bold green]PASSED[/bold green]"
            else:
                results['failed'] += 1
                status = "[bold red]FAILED[/bold red]"
            console.print(f"Test: {status}")
        
        results['details'].append(result)
    
    return results

def print_test_summary(results):
    """Print a summary of test results"""
    if not results:
        console.print("[bold red]No test results to display[/bold red]")
        return
    
    console.print("\n[bold blue]Test Summary[/bold blue]")
    console.print("=" * 50)
    
    total = results['total_tests']
    passed = results['passed']
    failed = results['failed']
    errors = results['errors']
    
    # Calculate percentages
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    console.print(f"Total Tests: {total}")
    console.print(f"Passed: [green]{passed}[/green] ({pass_rate:.1f}%)")
    console.print(f"Failed: [red]{failed}[/red]")
    console.print(f"Errors: [yellow]{errors}[/yellow]")
    
    # Export results to CSV
    df = pd.DataFrame(results['details'])
    df.to_csv('test_results.csv', index=False)
    console.print("\n[green]Test results exported to 'test_results.csv'[/green]")

def main():
    print("Incari Node Sequence Generator")
    print("Type 'quit' to exit or 'test' to run tests")
    print("-" * 50)
    
    models_llm = ['qwen2.5-coder:7b', 'qwen2.5-coder:14b', 'llama3.1:8b', 'codegemma:7b']
    selected_model = models_llm[0]
    
    # Print available models
    print("\nAvailable Models:")
    for i, model in enumerate(models_llm):
        print(f"{i+1}. {model}")
    print(f"\nCurrently using: {selected_model}")
    
    max_attempts = 10
    validation_threshold = 5
    
    while True:
        user_input = input("\nDescribe what you want to achieve (or type 'model <number>' to switch models): ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        elif user_input.lower().startswith('model '):
            try:
                model_num = int(user_input.split()[1]) - 1
                if 0 <= model_num < len(models_llm):
                    selected_model = models_llm[model_num]
                    print(f"\nSwitched to model: {selected_model}")
                else:
                    print("\nInvalid model number. Please choose from the available models.")
            except (IndexError, ValueError):
                print("\nInvalid input. Use 'model <number>' to switch models.")
            continue
        elif user_input.lower() == 'test':
            console.print("[bold blue]Starting automated tests...[/bold blue]")
            console.print(f"[yellow]Using model: {selected_model}[/yellow]")
            test_cases = load_test_cases()
            if test_cases:
                results = run_tests(test_cases, selected_model)
                print_test_summary(results)
            continue
        elif not user_input:
            continue
            
        node_sequence, was_validated, debug_output = process_query(
            user_input, 
            max_attempts, 
            validation_threshold,
            selected_model,
            silent=False
        )
        if node_sequence:
            console.print(debug_output)
            console.print("\n[bold green]Final Node Sequence:[/bold green]")
            console.print(node_sequence)
            if not was_validated:
                console.print("[bold yellow]⚠️ Note: This sequence did not pass full validation but was the most frequent result[/bold yellow]")

if __name__ == "__main__":
    main()