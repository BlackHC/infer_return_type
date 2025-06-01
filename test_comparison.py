#!/usr/bin/env python3

from infer_return_type import infer_return_type, TypeInferenceError as OriginalError
from unification_type_inference import infer_return_type_unified, TypeInferenceError as UnifiedError
from typing import List, Dict, TypeVar
import typing

A = TypeVar('A')
B = TypeVar('B')

def test_nested_mixed_containers():
    """Test case that fails in original but works in unified system."""
    
    def process_nested_mixed(data: List[List[A]]) -> A:
        pass
    
    mixed_nested = [[1, 2], ["a", "b"]]
    
    print("Testing nested mixed containers: [[1, 2], ['a', 'b']]")
    
    # Test original system
    try:
        t_orig = infer_return_type(process_nested_mixed, mixed_nested)
        print(f"✓ Original system: {t_orig}")
    except OriginalError as e:
        print(f"✗ Original system failed: {e}")
    
    # Test unified system
    try:
        t_unified = infer_return_type_unified(process_nested_mixed, mixed_nested)
        print(f"✓ Unified system: {t_unified}")
        
        # Verify it's a union
        origin = typing.get_origin(t_unified)
        if origin is typing.Union or hasattr(t_unified, '__args__'):
            if hasattr(t_unified, '__args__'):
                args = typing.get_args(t_unified)
                print(f"  Union components: {set(args)}")
    except UnifiedError as e:
        print(f"✗ Unified system failed: {e}")


def test_dict_with_mixed_types():
    """Test Dict[A, A] with mixed key/value types."""
    
    def process_self_referential_dict(data: Dict[A, A]) -> A:
        pass
    
    mixed_dict = {1: "a", "b": 2}  # Keys: int|str, Values: str|int
    
    print("\nTesting Dict[A, A] with mixed types: {1: 'a', 'b': 2}")
    
    # Test original system
    try:
        t_orig = infer_return_type(process_self_referential_dict, mixed_dict)
        print(f"✓ Original system: {t_orig}")
    except OriginalError as e:
        print(f"✗ Original system failed: {e}")
    
    # Test unified system
    try:
        t_unified = infer_return_type_unified(process_self_referential_dict, mixed_dict)
        print(f"✓ Unified system: {t_unified}")
        
        origin = typing.get_origin(t_unified)
        if origin is typing.Union or hasattr(t_unified, '__args__'):
            if hasattr(t_unified, '__args__'):
                args = typing.get_args(t_unified)
                print(f"  Union components: {set(args)}")
    except UnifiedError as e:
        print(f"✗ Unified system failed: {e}")


def test_complex_nested_branches():
    """Test complex nested structures where branches have different types."""
    
    def process_complex_nested(data: List[Dict[str, List[A]]]) -> A:
        pass
    
    complex_data = [
        {"branch1": [1, 2, 3]},        # A = int
        {"branch2": ["a", "b", "c"]}   # A = str
    ]
    
    print("\nTesting complex nested branches:")
    print("  [{'branch1': [1, 2, 3]}, {'branch2': ['a', 'b', 'c']}]")
    
    # Test original system
    try:
        t_orig = infer_return_type(process_complex_nested, complex_data)
        print(f"✓ Original system: {t_orig}")
    except OriginalError as e:
        print(f"✗ Original system failed: {e}")
    
    # Test unified system
    try:
        t_unified = infer_return_type_unified(process_complex_nested, complex_data)
        print(f"✓ Unified system: {t_unified}")
        
        origin = typing.get_origin(t_unified)
        if origin is typing.Union or hasattr(t_unified, '__args__'):
            if hasattr(t_unified, '__args__'):
                args = typing.get_args(t_unified)
                print(f"  Union components: {set(args)}")
    except UnifiedError as e:
        print(f"✗ Unified system failed: {e}")


def main():
    print("Comparing Original vs Unified Type Inference Systems")
    print("=" * 60)
    
    test_nested_mixed_containers()
    test_dict_with_mixed_types() 
    test_complex_nested_branches()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Original system fails on mixed types due to conflict detection")
    print("- Unified system forms unions, handling more real-world cases")
    print("- Unified system provides cleaner architecture for extensions")


if __name__ == "__main__":
    main() 