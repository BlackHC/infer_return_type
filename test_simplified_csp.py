"""Test script showing the simplified CSP approach using generic_utils."""

from csp_type_inference import infer_return_type_csp
from generic_utils import get_instance_concrete_args
from typing import TypeVar, List, Dict, Set

A = TypeVar('A')
B = TypeVar('B')

def test_generic_utils_integration():
    """Show how CSP now leverages generic_utils instead of manual type extraction."""
    
    print("=== Demonstrating generic_utils integration ===\n")
    
    # Show what generic_utils.get_instance_concrete_args returns
    mixed_list = [1, 'hello', 3.14]
    list_args = get_instance_concrete_args(mixed_list)
    print(f"get_instance_concrete_args([1, 'hello', 3.14]) = {list_args}")
    
    mixed_dict = {'a': 1, 'b': 'hello'}
    dict_args = get_instance_concrete_args(mixed_dict)
    print(f"get_instance_concrete_args({{'a': 1, 'b': 'hello'}}) = {dict_args}")
    
    mixed_set = {1, 'hello', 3.14}
    set_args = get_instance_concrete_args(mixed_set)
    print(f"get_instance_concrete_args({{1, 'hello', 3.14}}) = {set_args}")
    
    print("\n" + "="*50)
    print("CSP now uses these pre-computed types directly!")
    print("="*50 + "\n")
    
    # Show CSP results
    def list_example(data: List[A]) -> A: pass
    result = infer_return_type_csp(list_example, mixed_list)
    print(f"List[A] with mixed types: {result}")
    
    def dict_example(data: Dict[A, B]) -> A: pass  
    result = infer_return_type_csp(dict_example, mixed_dict)
    print(f"Dict[A, B] keys: {result}")
    
    def set_example(data: Set[A]) -> A: pass
    result = infer_return_type_csp(set_example, mixed_set)
    print(f"Set[A] with mixed types: {result}")

if __name__ == "__main__":
    test_generic_utils_integration() 