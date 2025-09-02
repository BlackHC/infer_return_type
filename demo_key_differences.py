#!/usr/bin/env python3
"""
Demo script showing key differences between the three infer_return_type implementations.
"""

import typing
from typing import TypeVar, List, Dict, Optional, Union, Callable
from dataclasses import dataclass

# Import all three implementations
from infer_return_type import infer_return_type as original_infer
from csp_type_inference import infer_return_type_csp as csp_infer
from unification_type_inference import infer_return_type_unified as unified_infer

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B')
Number = TypeVar('Number', int, float)  # Constrained
Numeric = TypeVar('Numeric', bound=float)  # Bounded


def print_result(impl_name: str, test_name: str, fn, args, kwargs=None):
    """Helper to print test results."""
    if kwargs is None:
        kwargs = {}
    
    try:
        if impl_name == "Original":
            result = original_infer(fn, *args, **kwargs)
        elif impl_name == "CSP":
            result = csp_infer(fn, *args, **kwargs)
        else:  # Unified
            result = unified_infer(fn, *args, **kwargs)
        print(f"  {impl_name:10} ✓ {result}")
    except Exception as e:
        print(f"  {impl_name:10} ✗ {type(e).__name__}: {str(e)[:60]}...")


def demo_conflict_handling():
    """Demo: How each implementation handles conflicting TypeVar bindings."""
    print("\n1. CONFLICT HANDLING")
    print("=" * 60)
    print("Function: identity(a: A, b: A) -> A")
    print("Call: identity(1, 'x')")
    print("-" * 60)
    
    def identity(a: A, b: A) -> A: ...
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "conflict", identity, [1, 'x'])
    
    print("\nKey insight: Original and CSP create unions, Unified fails")


def demo_bounded_typevar():
    """Demo: How bounded TypeVars are handled."""
    print("\n2. BOUNDED TYPEVAR HANDLING")
    print("=" * 60)
    print("TypeVar: Numeric = TypeVar('Numeric', bound=float)")
    print("Function: process_numeric(x: Numeric) -> Numeric")
    print("Call: process_numeric(1)  # int is NOT a subtype of float!")
    print("-" * 60)
    
    def process_numeric(x: Numeric) -> Numeric: ...
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "bounded", process_numeric, [1])
    
    print("\nKey insight: Only Unified correctly enforces the bound")


def demo_constrained_typevar():
    """Demo: How constrained TypeVars are handled."""
    print("\n3. CONSTRAINED TYPEVAR HANDLING")
    print("=" * 60)
    print("TypeVar: Number = TypeVar('Number', int, float)")
    print("Function: process_number(x: Number) -> Number")
    print("Call: process_number('not a number')")
    print("-" * 60)
    
    def process_number(x: Number) -> Number: ...
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "constrained", process_number, ['not a number'])
    
    print("\nKey insight: Original ignores constraints, CSP and Unified enforce them")


def demo_none_handling():
    """Demo: How None values are handled in Optional contexts."""
    print("\n4. NONE VALUE HANDLING")
    print("=" * 60)
    print("Function: process_dict_with_nones(d: Dict[str, Optional[A]]) -> A")
    print("Call: process_dict_with_nones({'a': 1, 'b': None, 'c': 2})")
    print("-" * 60)
    
    def process_dict_with_nones(d: Dict[str, Optional[A]]) -> A: ...
    
    test_dict = {'a': 1, 'b': None, 'c': 2}
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "none_handling", process_dict_with_nones, [test_dict])
    
    print("\nKey insight: Unified includes None in the result type")


def demo_complex_unions():
    """Demo: How complex union structures are handled."""
    print("\n5. COMPLEX UNION STRUCTURES")
    print("=" * 60)
    print("Function: extract_value(data: Dict[str, Union[A, List[A], Dict[str, A]]]) -> A")
    print("Call: extract_value({'single': 42, 'list': [43, 44], 'nested': {'value': 45}})")
    print("-" * 60)
    
    def extract_value(data: Dict[str, Union[A, List[A], Dict[str, A]]]) -> A: ...
    
    test_data = {
        'single': 42,
        'list': [43, 44],
        'nested': {'value': 45}
    }
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "complex_union", extract_value, [test_data])
    
    print("\nKey insight: CSP handles complex nested unions best")


def demo_mixed_containers():
    """Demo: How mixed-type containers are handled."""
    print("\n6. MIXED TYPE CONTAINERS")
    print("=" * 60)
    print("Function: process_list(items: List[A]) -> A")
    print("Call: process_list([1, 'hello', 3.14])")
    print("-" * 60)
    
    def process_list(items: List[A]) -> A: ...
    
    mixed_list = [1, 'hello', 3.14]
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "mixed_list", process_list, [mixed_list])
    
    print("\nKey insight: All implementations create union types")


def demo_empty_containers():
    """Demo: How empty containers are handled."""
    print("\n7. EMPTY CONTAINER HANDLING")
    print("=" * 60)
    print("Function: head(xs: List[A]) -> A")
    print("Call: head([])  # No type information!")
    print("-" * 60)
    
    def head(xs: List[A]) -> A: ...
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "empty", head, [[]])
    
    print("\nWith type_overrides={A: int}:")
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "empty_override", head, [[]], {'type_overrides': {A: int}})
    
    print("\nKey insight: All require type_overrides for empty containers")


def demo_variance():
    """Demo: How variance is handled in complex scenarios."""
    print("\n8. VARIANCE IN PRACTICE")
    print("=" * 60)
    print("Function: process_containers(l: List[A], d: Dict[A, str]) -> A")
    print("Call: process_containers([1, 2], {'x': 'hello'})  # Different types!")
    print("-" * 60)
    
    def process_containers(l: List[A], d: Dict[A, str]) -> A: ...
    
    for impl in ["Original", "CSP", "Unified"]:
        print_result(impl, "variance", process_containers, [[1, 2], {'x': 'hello'}])
    
    print("\nKey insight: Different approaches to variance create different results")


def main():
    """Run all demos."""
    print("COMPARISON OF THREE infer_return_type IMPLEMENTATIONS")
    print("=" * 60)
    
    demo_conflict_handling()
    demo_bounded_typevar()
    demo_constrained_typevar()
    demo_none_handling()
    demo_complex_unions()
    demo_mixed_containers()
    demo_empty_containers()
    demo_variance()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("• Original: Simple, lenient, creates unions on conflicts")
    print("• CSP: Sophisticated constraint solver, best for complex cases")
    print("• Unified: Strict, principled, enforces type system rules")
    print("\nChoose based on your needs:")
    print("- Need flexibility? → CSP")
    print("- Need strictness? → Unified")
    print("- Need simplicity? → Original")


if __name__ == "__main__":
    main()
