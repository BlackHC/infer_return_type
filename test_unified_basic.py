#!/usr/bin/env python3

from unification_type_inference import infer_return_type_unified
from typing import List, TypeVar
import typing

A = TypeVar('A')

def process_list(items: List[A]) -> A:
    pass

def main():
    print("Testing unification-based type inference...")
    
    # Test basic homogeneous case
    try:
        t1 = infer_return_type_unified(process_list, [1, 2, 3])
        print(f"✓ Homogeneous list [1, 2, 3]: {t1}")
    except Exception as e:
        print(f"✗ Homogeneous list failed: {e}")
    
    # Test mixed types (should form union instead of failing)
    try:
        t2 = infer_return_type_unified(process_list, [1, 'hello'])
        print(f"✓ Mixed types [1, 'hello']: {t2}")
        
        # Check if it's a union
        origin = typing.get_origin(t2)
        if origin is typing.Union:
            args = typing.get_args(t2)
            print(f"  Union args: {args}")
        
    except Exception as e:
        print(f"✗ Mixed types failed: {e}")

if __name__ == "__main__":
    main() 