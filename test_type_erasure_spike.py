"""
Spike tests to verify Python's type annotation behavior vs type erasure.

These tests verify that function annotations like `list[dict[str, list[int]]]`
preserve full nested generic type information at runtime, while actual instances
undergo type erasure.

CONCLUSIONS:
1. Function annotations preserve full nested generic types for built-in types
2. Runtime instances undergo type erasure (lose generic parameters)
3. This applies to both modern and legacy typing syntax
4. IMPORTANT: Different storage mechanisms for custom generic TypeVar info:
   - Pydantic generics: TypeVar info stored in class.__pydantic_generic_metadata__['parameters']
   - Dataclass generics: TypeVar info preserved directly in annotation get_args()
5. Both custom types preserve concrete type info in instances:
   - Pydantic: __pydantic_generic_metadata__['args'] on instances
   - Dataclass: __orig_class__ on instances (when explicitly typed)
6. KEY INSIGHT: TypeVar information is preserved in annotations for both, just accessed differently!
"""

import inspect
from typing import get_origin, get_args, List, Dict, TypeVar, Union
from dataclasses import dataclass
import typing


def _get_annotation(func, param_name="arg"):
    """Helper to extract parameter annotation from function."""
    sig = inspect.signature(func)
    return sig.parameters[param_name].annotation


def _analyze_nested_generics(annotation, expected_structure):
    """Helper to verify nested generic structure matches expectations."""
    current = annotation
    for expected_origin, *args in expected_structure:
        assert (
            get_origin(current) is expected_origin
        ), f"Expected {expected_origin}, got {get_origin(current)}"
        type_args = get_args(current)
        if args:
            current = type_args[args[0]]  # Move to specified argument index
    return True


def test_annotation_preservation():
    """Test that function annotations preserve full nested generic type information."""

    def test_func(arg: list[dict[str, list[int]]]) -> dict[str, list[tuple[int, str]]]:
        pass

    # Test parameter annotation
    param_annotation = _get_annotation(test_func)
    expected_param_structure = [
        (list, 0),  # list[...] -> dict[...]
        (dict, 1),  # dict[str, ...] -> list[...]
        (list, 0),  # list[...] -> int
    ]
    _analyze_nested_generics(param_annotation, expected_param_structure)

    # Test return annotation
    return_annotation = inspect.signature(test_func).return_annotation
    expected_return_structure = [
        (dict, 1),  # dict[str, ...] -> list[...]
        (list, 0),  # list[...] -> tuple[...]
        (tuple,),  # tuple[int, str]
    ]
    _analyze_nested_generics(return_annotation, expected_return_structure)

    # Verify final types
    dict_args = get_args(param_annotation)[0]  # dict[str, list[int]]
    assert get_args(dict_args) == (str, list[int])

    print(f"✓ Parameter annotation preserved: {param_annotation}")
    print(f"✓ Return annotation preserved: {return_annotation}")


def test_instance_type_erasure():
    """Test that runtime instances undergo type erasure."""
    # Create instances matching the annotation types
    nested_data: list[dict[str, list[int]]] = [{"numbers": [1, 2, 3]}]

    # Instances lose all generic type information
    assert type(nested_data) is list
    assert get_args(type(nested_data)) == ()

    assert type(nested_data[0]) is dict
    assert get_args(type(nested_data[0])) == ()

    assert type(nested_data[0]["numbers"]) is list
    assert get_args(type(nested_data[0]["numbers"])) == ()

    print(
        f"✓ Instance types erased: {type(nested_data)} -> {type(nested_data[0])} -> {type(nested_data[0]['numbers'])}"
    )


def test_legacy_vs_modern_syntax():
    """Test that both legacy (typing.List) and modern (list) syntax preserve annotations."""

    def modern_func(arg: list[dict[str, list[int]]]) -> None:
        pass

    def legacy_func(arg: List[Dict[str, List[int]]]) -> None:
        pass

    modern_annotation = _get_annotation(modern_func)
    legacy_annotation = _get_annotation(legacy_func)

    # Both preserve structure, just different string representations
    assert str(modern_annotation) == "list[dict[str, list[int]]]"
    assert "List" in str(legacy_annotation) and "Dict" in str(legacy_annotation)

    # Both have same underlying structure - get_origin() returns builtin types for both
    _analyze_nested_generics(modern_annotation, [(list, 0), (dict,)])
    _analyze_nested_generics(
        legacy_annotation, [(list, 0), (dict,)]
    )  # get_origin(typing.List) returns list

    print(f"✓ Modern syntax: {modern_annotation}")
    print(f"✓ Legacy syntax: {legacy_annotation}")


def test_typevar_annotation_preservation():
    """Test that function annotations preserve TypeVar information."""
    # Define TypeVars
    A = TypeVar("A")
    B = TypeVar("B")

    def test_func(arg: list[dict[A, B]]) -> set[A | B]:
        pass

    # Test parameter annotation preserves TypeVar structure
    param_annotation = _get_annotation(test_func)

    # Verify outer structure: list[dict[A, B]]
    assert get_origin(param_annotation) is list
    dict_type = get_args(param_annotation)[0]
    assert get_origin(dict_type) is dict

    # Verify TypeVars are preserved in dict type args
    dict_args = get_args(dict_type)
    assert len(dict_args) == 2
    assert dict_args[0] is A  # First TypeVar preserved
    assert dict_args[1] is B  # Second TypeVar preserved

    # Test return annotation preserves Union of TypeVars
    return_annotation = inspect.signature(test_func).return_annotation
    assert get_origin(return_annotation) is set

    union_type = get_args(return_annotation)[0]
    assert get_origin(union_type) is Union
    union_args = get_args(union_type)
    assert len(union_args) == 2
    assert A in union_args and B in union_args

    print(f"✓ TypeVar parameter annotation preserved: {param_annotation}")
    print(f"✓ TypeVar return annotation preserved: {return_annotation}")
    print(f"✓ TypeVars A and B preserved in annotations: {dict_args}")


def test_typevar_instance_type_erasure():
    """Test that runtime instances with TypeVars still undergo type erasure."""
    A = TypeVar("A")
    B = TypeVar("B")

    # Create instances - concrete types lose TypeVar information
    nested_data: list[dict[A, B]] = [{"key": "value"}]  # A=str, B=str at runtime
    result_set: set[A | B] = {"item1", "item2"}  # A|B=str at runtime

    # Instances lose all generic and TypeVar information
    assert type(nested_data) is list
    assert get_args(type(nested_data)) == ()

    assert type(nested_data[0]) is dict
    assert get_args(type(nested_data[0])) == ()

    assert type(result_set) is set
    assert get_args(type(result_set)) == ()

    print(
        f"✓ TypeVar instance types erased: {type(nested_data)} -> {type(nested_data[0])} -> {type(result_set)}"
    )


def test_bound_typevar_annotations():
    """Test TypeVars with bounds preserve bound information in annotations."""
    T = TypeVar("T", bound=str)
    U = TypeVar("U", int, float)  # Constrained TypeVar

    def test_func(arg: list[list[T]]) -> dict[str, U]:
        pass

    param_annotation = _get_annotation(test_func)
    return_annotation = inspect.signature(test_func).return_annotation

    # Verify TypeVar with bound is preserved
    list_arg = get_args(param_annotation)[0]
    list_arg = get_args(list_arg)[0]
    assert list_arg is T
    assert getattr(list_arg, "__bound__", None) is str

    # Verify constrained TypeVar is preserved
    dict_value_type = get_args(return_annotation)[1]
    assert dict_value_type is U
    assert getattr(dict_value_type, "__constraints__", ()) == (int, float)

    print(f"✓ Bound TypeVar preserved: {T} with bound {getattr(T, '__bound__', None)}")
    print(
        f"✓ Constrained TypeVar preserved: {U} with constraints {getattr(U, '__constraints__', ())}"
    )


def test_custom_generic_partial_annotation_erasure():
    """
    Test annotation behavior with custom generic types.

    Investigates whether TypeVar information is truly lost or just stored differently
    in Pydantic vs dataclass annotations.
    """
    from pydantic import BaseModel

    A = TypeVar("A")

    # Define a custom generic type
    class CustomBox(BaseModel, typing.Generic[A]):
        item: A

    @dataclass
    class CustomWrap(typing.Generic[A]):
        value: A

    # Test function with nested custom generics
    def process_boxes(boxes: List[CustomBox[A]]) -> List[A]:
        pass

    def process_wraps(wraps: List[CustomWrap[A]]) -> List[A]:
        pass

    # Check annotations - this reveals the difference
    boxes_annotation = _get_annotation(process_boxes, "boxes")
    wraps_annotation = _get_annotation(process_wraps, "wraps")

    print(f"✓ Boxes annotation: {boxes_annotation}")
    print(f"✓ Wraps annotation: {wraps_annotation}")

    # Analyze Pydantic annotation behavior
    assert get_origin(boxes_annotation) is list
    box_type = get_args(boxes_annotation)[0]
    print(f"✓ Box type in annotation: {box_type}")
    print(f"  - get_args(box_type): {get_args(box_type)}")

    # Check if the Pydantic class itself preserves TypeVar info
    if hasattr(box_type, "__pydantic_generic_metadata__"):
        box_metadata = box_type.__pydantic_generic_metadata__
        print(f"  - Box class __pydantic_generic_metadata__: {box_metadata}")

        # Check if TypeVar info is in the parameters field
        if "parameters" in box_metadata:
            print(f"  - Parameters: {box_metadata['parameters']}")
            # Verify if our TypeVar A is in there
            if box_metadata["parameters"] and A in box_metadata["parameters"]:
                print(f"  - ✓ TypeVar A found in parameters!")
            else:
                print(f"  - ✗ TypeVar A not found in parameters")

        if "args" in box_metadata:
            print(f"  - Args: {box_metadata['args']}")

    else:
        print(f"  - Box class has no __pydantic_generic_metadata__")

    # Compare with dataclass behavior
    assert get_origin(wraps_annotation) is list
    wrap_type = get_args(wraps_annotation)[0]
    print(f"✓ Wrap type in annotation: {wrap_type}")
    print(f"  - get_origin(wrap_type): {get_origin(wrap_type)}")
    print(f"  - get_args(wrap_type): {get_args(wrap_type)}")

    # Verify dataclass preserves TypeVar in annotation structure
    if get_args(wrap_type):
        assert get_args(wrap_type) == (A,)  # TypeVar info preserved in annotation!
        print(f"  - ✓ TypeVar A preserved directly in annotation args")

    # Test with instances to see concrete type preservation
    box_instance = CustomBox[int](item=42)
    wrap_instance = CustomWrap[str]("hello")

    # Pydantic instance metadata
    box_instance_metadata = getattr(box_instance, "__pydantic_generic_metadata__", {})
    print(f"✓ Box instance metadata: {box_instance_metadata}")
    if "args" in box_instance_metadata:
        assert box_instance_metadata["args"] == (int,)
        print(f"  - ✓ Concrete type (int) preserved in instance")

    # Dataclass instance __orig_class__
    if hasattr(wrap_instance, "__orig_class__"):
        print(f"✓ Wrap instance __orig_class__: {wrap_instance.__orig_class__}")
        assert wrap_instance.__orig_class__ == CustomWrap[str]
        assert get_args(wrap_instance.__orig_class__) == (str,)
        print(f"  - ✓ Concrete type (str) preserved in __orig_class__")
    else:
        print(f"✗ Wrap instance has no __orig_class__")
        assert False

    print(f"\n✓ FINDINGS:")
    print(
        f"  - Pydantic: TypeVar info preserved in class.__pydantic_generic_metadata__['parameters']"
    )
    print(f"  - Dataclass: TypeVar info preserved directly in annotation get_args()")
    print(f"  - Both preserve concrete types in instances via different mechanisms")
    print(
        f"✓ This means type inference can recover TypeVar info from both annotation approaches!"
    )


if __name__ == "__main__":
    print("Testing Python type annotation preservation vs type erasure...")
    print()

    test_annotation_preservation()
    print()

    test_instance_type_erasure()
    print()

    test_legacy_vs_modern_syntax()
    print()

    test_typevar_annotation_preservation()
    print()

    test_typevar_instance_type_erasure()
    print()

    test_bound_typevar_annotations()
    print()

    test_custom_generic_partial_annotation_erasure()
    print()

    print("CONCLUSION:")
    print("✓ Function annotations preserve full nested generic types")
    print("✓ Runtime instances undergo type erasure (lose generic parameters)")
    print("✓ This applies to both modern and legacy typing syntax")
    print("✓ CORRECTED: Custom generics preserve TypeVar info differently:")
    print("  - Pydantic: class.__pydantic_generic_metadata__['parameters']")
    print("  - Dataclass: annotation get_args() directly")
    print("✓ Both custom generic instances preserve concrete type info")
