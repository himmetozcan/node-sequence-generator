import unittest
from node_seq_gen import (
    process_query, 
    validate_node_names,
    load_test_cases
)
import time

class TestNodeSeqSystem(unittest.TestCase):
    def setUp(self):
        self.model = "qwen2.5-coder:7b"
        
    def test_process_query_basic(self):
        """Test basic query processing"""
        query = "When a button is clicked, fetch data and display it in a modal"
        result, validated, debug_output = process_query(
            query,  
            max_attempts=7,
            validation_threshold=4,
            selected_model=self.model,
            silent=True
        )
        # Just verify we got a valid sequence
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
    def test_validate_node_names(self):
        """Test node name validation"""
        valid_sequence = ["OnClick", "FetchData", "DisplayModal"]
        invalid_sequence = ["OnClick", "InvalidNode", "DisplayModal"]
        
        # Test valid sequence
        is_valid, invalid_nodes = validate_node_names(valid_sequence)
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_nodes), 0)
        
        # Test invalid sequence
        is_valid, invalid_nodes = validate_node_names(invalid_sequence)
        self.assertFalse(is_valid)
        self.assertEqual(invalid_nodes, ["InvalidNode"])

            
    def test_complex_queries(self):
        """Test processing of more complex queries"""
        test_cases = load_test_cases(max_tests=100)  # Load all test cases from JSON file
        if not test_cases:
            self.fail("Could not load test cases from JSON file")
        
        total_cases = len(test_cases)
        passed_cases = 0
        total_time = 0
        
        for case in test_cases:
            start_time = time.time()
            result, validated, debug_output = process_query(
                case["User Prompt"],
                max_attempts=3,
                validation_threshold=2,
                selected_model=self.model,
                silent=True
            )
            total_time += time.time() - start_time
            
            if result == case["Correct Output"]:
                passed_cases += 1
        
        success_rate = (passed_cases / total_cases) * 100
        avg_time = total_time / total_cases
        
        print(f"\nTest Summary:")
        print(f"Total Cases: {total_cases}")
        print(f"Passed: {passed_cases} ({success_rate:.2f}%)")
        print(f"Average Time per Case: {avg_time:.2f}s")
        
            
if __name__ == '__main__':
    unittest.main() 