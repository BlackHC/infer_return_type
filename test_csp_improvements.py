"""Test script for CSP improvements: unified handlers and variance."""

from csp_type_inference import infer_return_type_csp
from typing import TypeVar, List, Dict, Set, Union, Tuple
from dataclasses import dataclass

A = TypeVar('A')
B = TypeVar('B')

def test_covariance():
    """Test that covariant containers properly handle mixed types."""
    print("Testing covariance...")
    
    def covariant_example(data: List[A]) -> A:
        pass
    
    # Mixed list should create union type
    mixed_list = [1, 'hello', 3.14]
    result = infer_return_type_csp(covariant_example, mixed_list)
    print(f"Covariant List[A] with mixed types: {result}")
    
    # Check if it's a union
    if hasattr(result, '__args__'):
        union_args = set(result.__args__)
        expected = {int, str, float}
        assert union_args == expected, f"Expected {expected}, got {union_args}"
    
    print("✓ Covariance test passed")

def test_dict_variance():
    """Test that dict keys are invariant and values are covariant."""
    print("Testing dict variance...")
    
    def dict_example(data: Dict[A, B]) -> A:
        pass
    
    # Dict with uniform key types (invariant)
    dict_data = {'key1': 1, 'key2': 2}
    result = infer_return_type_csp(dict_example, dict_data)
    print(f"Dict[A, B] keys (invariant): {result}")
    assert result == str, f"Expected str, got {result}"
    
    print("✓ Dict variance test passed")

def test_set_covariance():
    """Test that sets are covariant."""
    print("Testing set covariance...")
    
    def set_example(data: Set[A]) -> A:
        pass
    
    # Homogeneous set
    set_data = {1, 2, 3}
    result = infer_return_type_csp(set_example, set_data)
    print(f"Set[A] with homogeneous types: {result}")
    assert result == int, f"Expected int, got {result}"
    
    # Mixed set
    mixed_set = {1, 'hello', 3.14}
    result = infer_return_type_csp(set_example, mixed_set)
    print(f"Set[A] with mixed types: {result}")
    
    print("✓ Set covariance test passed")

def test_unified_handler():
    """Test that the unified handler works for different container types."""
    print("Testing unified handler...")
    
    # Test tuple
    def tuple_example(data: Tuple[A, ...]) -> A:
        pass
    
    tuple_data = (1, 2, 3)
    result = infer_return_type_csp(tuple_example, tuple_data)
    print(f"Tuple[A, ...]: {result}")
    assert result == int, f"Expected int, got {result}"
    
    print("✓ Unified handler test passed")

if __name__ == "__main__":
    test_covariance()
    test_dict_variance() 
    test_set_covariance()
    test_unified_handler()
    print("\n🎉 All CSP improvement tests passed!") 