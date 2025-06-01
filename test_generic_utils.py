"""
Unit tests for generic_utils module.

Tests the unified interface for extracting type information from different
generic type systems (built-ins, Pydantic, dataclasses).
"""

import pytest
import typing
import types
from typing import TypeVar, List, Dict, Tuple, Set, Union, get_origin, get_args
from dataclasses import dataclass

from generic_utils import (
    GenericInfo, GenericTypeUtils, BuiltinExtractor, PydanticExtractor, DataclassExtractor,
    UnionExtractor, create_union_if_needed,
    get_generic_info, get_instance_generic_info, get_type_parameters, get_concrete_args,
    get_instance_concrete_args, get_generic_origin, is_generic_type, extract_all_typevars
)

# Test fixtures
A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T', bound=str)
U = TypeVar('U', int, float)


def _is_union_type(obj):
    """Helper to check if object is a Union type (handles both typing.Union and types.UnionType)."""
    origin = get_origin(obj)
    return origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType'))


# Pydantic test classes (conditional import for testing)
try:
    from pydantic import BaseModel
    
    class PydanticBox(BaseModel, typing.Generic[A]):
        item: A
    
    class PydanticPair(BaseModel, typing.Generic[A, B]):
        first: A
        second: B
    
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


# Dataclass test classes
@dataclass
class DataclassBox(typing.Generic[A]):
    item: A


@dataclass 
class DataclassPair(typing.Generic[A, B]):
    first: A
    second: B


class TestBuiltinExtractor:
    """Test built-in generic type extraction."""
    
    def test_list_annotation(self):
        extractor = BuiltinExtractor()
        
        # Non-generic list
        assert not extractor.can_handle_annotation(list)
        
        # Generic list with concrete type
        info = extractor.extract_from_annotation(list[int])
        assert info.origin is list
        assert info.concrete_args == [int]
        assert info.type_params == []
        assert info.is_generic
        
        # Generic list with TypeVar
        info = extractor.extract_from_annotation(list[A])
        assert info.origin is list
        assert info.concrete_args == [A]
        assert info.type_params == [A]
        assert info.is_generic
    
    def test_dict_annotation(self):
        extractor = BuiltinExtractor()
        
        info = extractor.extract_from_annotation(dict[str, int])
        assert info.origin is dict
        assert info.concrete_args == [str, int]
        assert info.type_params == []
        assert info.is_generic
        
        info = extractor.extract_from_annotation(dict[A, B])
        assert info.origin is dict
        assert info.concrete_args == [A, B]
        assert info.type_params == [A, B]
        assert info.is_generic
    
    def test_tuple_annotation(self):
        extractor = BuiltinExtractor()
        
        info = extractor.extract_from_annotation(tuple[int, str, float])
        assert info.origin is tuple
        assert info.concrete_args == [int, str, float]
        assert info.type_params == []
        assert info.is_generic
        
        # Variable length tuple
        info = extractor.extract_from_annotation(tuple[A, ...])
        assert info.origin is tuple
        assert info.concrete_args == [A, ...]
        assert info.type_params == [A]
        assert info.is_generic
    
    def test_set_annotation(self):
        extractor = BuiltinExtractor()
        
        info = extractor.extract_from_annotation(set[int])
        assert info.origin is set
        assert info.concrete_args == [int]
        assert info.type_params == []
        assert info.is_generic
    
    def test_legacy_typing_annotations(self):
        extractor = BuiltinExtractor()
        
        info = extractor.extract_from_annotation(List[int])
        assert info.origin is list
        assert info.concrete_args == [int]
        assert info.is_generic
        
        info = extractor.extract_from_annotation(Dict[str, int])
        assert info.origin is dict
        assert info.concrete_args == [str, int]
        assert info.is_generic
    
    def test_list_instance(self):
        extractor = BuiltinExtractor()
        
        # Empty list
        info = extractor.extract_from_instance([])
        assert info.origin is list
        assert info.concrete_args == []
        assert not info.is_generic
        
        # Homogeneous list
        info = extractor.extract_from_instance([1, 2, 3])
        assert info.origin is list
        assert info.concrete_args == [int]
        assert info.is_generic
        
        # Mixed type list
        info = extractor.extract_from_instance([1, "hello"])
        assert info.origin is list
        assert len(info.concrete_args) == 1
        # Should be a union type (either typing.Union or types.UnionType)
        union_type = info.concrete_args[0]
        assert _is_union_type(union_type)
        assert set(get_args(union_type)) == {int, str}
        assert info.is_generic
    
    def test_dict_instance(self):
        extractor = BuiltinExtractor()
        
        # Empty dict
        info = extractor.extract_from_instance({})
        assert info.origin is dict
        assert info.concrete_args == []
        assert not info.is_generic
        
        # Homogeneous dict
        info = extractor.extract_from_instance({"a": 1, "b": 2})
        assert info.origin is dict
        assert info.concrete_args == [str, int]
        assert info.is_generic
        
        # Mixed type dict
        info = extractor.extract_from_instance({"a": 1, "b": "hello"})
        assert info.origin is dict
        assert len(info.concrete_args) == 2
        assert info.concrete_args[0] == str  # Keys are homogeneous
        union_type = info.concrete_args[1]  # Values are mixed
        assert _is_union_type(union_type)
        assert set(get_args(union_type)) == {int, str}
        assert info.is_generic
    
    def test_tuple_instance(self):
        extractor = BuiltinExtractor()
        
        info = extractor.extract_from_instance((1, "hello", 3.14))
        assert info.origin is tuple
        assert info.concrete_args == [int, str, float]
        assert info.is_generic
    
    def test_set_instance(self):
        extractor = BuiltinExtractor()
        
        # Empty set
        info = extractor.extract_from_instance(set())
        assert info.origin is set
        assert info.concrete_args == []
        assert not info.is_generic
        
        # Homogeneous set
        info = extractor.extract_from_instance({1, 2, 3})
        assert info.origin is set
        assert info.concrete_args == [int]
        assert info.is_generic
        
        # Mixed type set
        info = extractor.extract_from_instance({1, "hello"})
        assert info.origin is set
        assert len(info.concrete_args) == 1
        union_type = info.concrete_args[0]
        assert _is_union_type(union_type)
        assert set(get_args(union_type)) == {int, str}
        assert info.is_generic


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestPydanticExtractor:
    """Test Pydantic generic type extraction."""
    
    def test_pydantic_annotation(self):
        extractor = PydanticExtractor()
        
        # Generic Pydantic class
        assert extractor.can_handle_annotation(PydanticBox)
        
        info = extractor.extract_from_annotation(PydanticBox)
        assert info.origin is PydanticBox
        assert A in info.type_params
        assert info.is_generic
        
        # Parameterized Pydantic class
        info = extractor.extract_from_annotation(PydanticBox[int])
        assert info.origin is PydanticBox  # Should be unparameterized base
        assert info.concrete_args == [int]
        assert info.is_generic
        
        # Generic Pydantic class with TypeVar
        info = extractor.extract_from_annotation(PydanticBox[B])
        assert info.origin is PydanticBox  # Should be unparameterized base
        assert info.concrete_args == [B]
        assert info.is_generic
    
    def test_pydantic_multi_param(self):
        extractor = PydanticExtractor()
        
        info = extractor.extract_from_annotation(PydanticPair)
        assert info.origin is PydanticPair
        assert A in info.type_params
        assert B in info.type_params
        assert len(info.type_params) == 2
        assert info.is_generic
    
    def test_pydantic_instance(self):
        extractor = PydanticExtractor()
        
        # Instance with concrete type
        instance = PydanticBox[int](item=42)
        
        assert extractor.can_handle_instance(instance)
        
        info = extractor.extract_from_instance(instance)
        # Should get the base class, not the parameterized class
        assert info.origin is PydanticBox or info.origin.__name__ == 'PydanticBox'
        assert info.concrete_args == [int]
        assert A in info.type_params  # From class metadata
        assert info.is_generic
    
    def test_pydantic_multi_param_instance(self):
        extractor = PydanticExtractor()
        
        instance = PydanticPair[str, int](first="hello", second=42)
        
        info = extractor.extract_from_instance(instance)
        # Should get the base class, not the parameterized class
        assert info.origin is PydanticPair or info.origin.__name__ == 'PydanticPair'
        assert info.concrete_args == [str, int]
        assert A in info.type_params
        assert B in info.type_params
        assert info.is_generic


class TestDataclassExtractor:
    """Test dataclass generic type extraction."""
    
    def test_dataclass_annotation(self):
        extractor = DataclassExtractor()
        
        # Generic dataclass
        assert extractor.can_handle_annotation(DataclassBox)
        
        info = extractor.extract_from_annotation(DataclassBox[int])
        assert info.origin is DataclassBox
        assert info.concrete_args == [int]
        assert info.type_params == []  # Concrete type, no TypeVars in current annotation
        assert info.is_generic
        
        # With TypeVar
        info = extractor.extract_from_annotation(DataclassBox[A])
        assert info.origin is DataclassBox
        assert info.concrete_args == [A]
        assert info.type_params == [A]  # TypeVar present in current annotation
        assert info.is_generic
    
    def test_dataclass_multi_param(self):
        extractor = DataclassExtractor()
        
        info = extractor.extract_from_annotation(DataclassPair[A, B])
        assert info.origin is DataclassPair
        assert info.concrete_args == [A, B]
        assert info.type_params == [A, B]
        assert info.is_generic
    
    def test_dataclass_instance(self):
        extractor = DataclassExtractor()
        
        # Instance with __orig_class__ 
        instance = DataclassBox[int](item=42)
        instance.__orig_class__ = DataclassBox[int]  # Simulate what Python does
        
        assert extractor.can_handle_instance(instance)
        
        info = extractor.extract_from_instance(instance)
        assert info.origin is DataclassBox
        assert info.concrete_args == [int]
        assert info.is_generic
    
    def test_dataclass_instance_without_orig_class(self):
        extractor = DataclassExtractor()
        
        # Instance without __orig_class__
        instance = DataclassBox(item=42)
        
        info = extractor.extract_from_instance(instance)
        assert info.origin is DataclassBox
        assert info.concrete_args == []  # Can't infer without __orig_class__
        # Still might be generic based on class definition
        assert info.is_generic  # Has TypeVar params from class


class TestGenericTypeUtils:
    """Test the unified interface."""
    
    def test_builtin_types(self):
        utils = GenericTypeUtils()
        
        # List
        info = utils.get_generic_info(list[int])
        assert info.origin is list
        assert info.concrete_args == [int]
        assert info.is_generic
        
        # TypeVars
        type_params = utils.get_type_parameters(list[A])
        assert A in type_params
        
        # Instance
        instance_args = utils.get_instance_concrete_args([1, 2, 3])
        assert instance_args == [int]
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_types(self):
        utils = GenericTypeUtils()
        
        # Annotation
        info = utils.get_generic_info(PydanticBox)
        assert info.origin is PydanticBox
        assert A in info.type_params
        
        # Instance
        instance = PydanticBox[str](item="hello")
        instance_args = utils.get_instance_concrete_args(instance)
        assert instance_args == [str]
    
    def test_dataclass_types(self):
        utils = GenericTypeUtils()
        
        # Annotation
        info = utils.get_generic_info(DataclassBox[A])
        assert info.origin is DataclassBox
        assert A in info.type_params
        
        # Instance with __orig_class__
        instance = DataclassBox[int](item=42)
        instance.__orig_class__ = DataclassBox[int]
        instance_args = utils.get_instance_concrete_args(instance)
        assert instance_args == [int]
    
    def test_non_generic_types(self):
        utils = GenericTypeUtils()
        
        # Regular class
        info = utils.get_generic_info(str)
        assert info.origin is str
        assert not info.is_generic
        assert info.type_params == []
        assert info.concrete_args == []
        
        # Regular instance
        info = utils.get_instance_generic_info("hello")
        assert info.origin is str
        assert not info.is_generic
    
    def test_extract_all_typevars(self):
        utils = GenericTypeUtils()
        
        # Simple case
        typevars = utils.extract_all_typevars(list[A])
        assert typevars == [A]
        
        # Nested case
        typevars = utils.extract_all_typevars(dict[A, list[B]])
        assert set(typevars) == {A, B}
        
        # Deep nesting
        typevars = utils.extract_all_typevars(list[dict[A, tuple[B, int]]])
        assert set(typevars) == {A, B}
        
        # No TypeVars
        typevars = utils.extract_all_typevars(list[int])
        assert typevars == []
    
    def test_union_types(self):
        utils = GenericTypeUtils()
        
        # Union with TypeVars
        typevars = utils.extract_all_typevars(Union[A, B])
        assert set(typevars) == {A, B}
        
        # Nested Union
        typevars = utils.extract_all_typevars(list[Union[A, int]])
        assert typevars == [A]


class TestConvenienceFunctions:
    """Test the module-level convenience functions."""
    
    def test_get_generic_info(self):
        info = get_generic_info(list[int])
        assert info.origin is list
        assert info.concrete_args == [int]
        assert info.is_generic
    
    def test_get_type_parameters(self):
        params = get_type_parameters(dict[A, B])
        assert set(params) == {A, B}
    
    def test_get_concrete_args(self):
        args = get_concrete_args(tuple[int, str])
        assert args == [int, str]
    
    def test_get_instance_concrete_args(self):
        args = get_instance_concrete_args([1, 2, 3])
        assert args == [int]
    
    def test_get_generic_origin(self):
        origin = get_generic_origin(list[int])
        assert origin is list
        
        origin = get_generic_origin(str)
        assert origin is str
    
    def test_is_generic_type(self):
        assert is_generic_type(list[int])
        assert is_generic_type(dict[A, B])
        assert not is_generic_type(str)
        assert not is_generic_type(int)
    
    def test_extract_all_typevars(self):
        typevars = extract_all_typevars(list[dict[A, B]])
        assert set(typevars) == {A, B}


class TestComplexScenarios:
    """Test complex nested scenarios."""
    
    def test_deeply_nested_builtin(self):
        # list[dict[str, tuple[int, float, set[A]]]]
        annotation = list[dict[str, tuple[int, float, set[A]]]]
        
        typevars = extract_all_typevars(annotation)
        assert typevars == [A]
        
        info = get_generic_info(annotation)
        assert info.origin is list
        assert info.concrete_args == [dict[str, tuple[int, float, set[A]]]]
        assert info.is_generic
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_nested_custom_types(self):
        # list[PydanticBox[A]]
        annotation = list[PydanticBox[A]]
        
        typevars = extract_all_typevars(annotation)
        assert A in typevars
        
        # Instance
        box_instance = PydanticBox[int](item=42)
        list_instance = [box_instance]
        
        list_args = get_instance_concrete_args(list_instance)
        # Should get the type of the box instance
        assert len(list_args) == 1
        assert isinstance(list_args[0], type)  # Should be a type
        # It should be the actual type of the instance
        assert list_args[0] is type(box_instance)
    
    def test_union_with_generics(self):
        # Union[list[A], dict[B, int]]
        annotation = Union[list[A], dict[B, int]]
        
        typevars = extract_all_typevars(annotation)
        assert set(typevars) == {A, B}
    
    def test_bound_typevars(self):
        # Verify that bound TypeVars are preserved
        params = get_type_parameters(list[T])
        assert params == [T]
        assert params[0].__bound__ is str
        
        params = get_type_parameters(dict[U, int])
        assert params == [U]
        assert params[0].__constraints__ == (int, float)
    
    def test_nested_typevar_extraction_builtin(self):
        """Test that nested TypeVars are properly extracted from built-in types."""
        
        # Test cases that were failing before the fix
        info = get_generic_info(list[list[A]])
        assert info.type_params == [A], f"Expected [A], got {info.type_params}"
        assert info.concrete_args == [list[A]]
        
        info = get_generic_info(dict[A, list[B]])
        assert set(info.type_params) == {A, B}, f"Expected {{A, B}}, got {set(info.type_params)}"
        assert info.concrete_args == [A, list[B]]
        
        # Triple nesting
        info = get_generic_info(list[dict[A, set[B]]])
        assert set(info.type_params) == {A, B}
        assert info.concrete_args == [dict[A, set[B]]]
    
    def test_improved_instance_type_inference(self):
        """Test that instance type inference properly handles nested generics."""
        
        # Test case that was failing before the fix: {"a": [1, 2]}
        info = get_instance_generic_info({"a": [1, 2]})
        assert info.origin is dict
        assert len(info.concrete_args) == 2
        assert info.concrete_args[0] == str
        # Should be list[int], not just list
        assert info.concrete_args[1] == list[int]
        
        # Nested list inference
        info = get_instance_generic_info([1, [2, 3]])
        assert info.origin is list
        # Should create a union of int and list[int]
        union_type = info.concrete_args[0]
        assert _is_union_type(union_type)
        union_args = get_args(union_type)
        assert int in union_args
        assert list[int] in union_args
        
        # Deeply nested dict
        nested_dict = {"level1": {"level2": [1, 2, 3]}}
        info = get_instance_generic_info(nested_dict)
        assert info.origin is dict
        assert info.concrete_args[0] == str  # Keys
        # Values should be dict[str, list[int]]
        assert info.concrete_args[1] == dict[str, list[int]]
    
    def test_mixed_container_instance_inference(self):
        """Test instance inference with mixed types in containers."""
        
        # Mixed list with nested structures
        mixed_data = [1, {"a": 2}, [3, 4]]
        info = get_instance_generic_info(mixed_data)
        assert info.origin is list
        union_type = info.concrete_args[0]
        assert _is_union_type(union_type)
        union_args = get_args(union_type)
        assert int in union_args
        assert dict[str, int] in union_args
        assert list[int] in union_args
    
    def test_dataclass_with_nested_generics(self):
        """Test dataclass with nested generic annotations."""
        
        @dataclass
        class NestedContainer(typing.Generic[A, B]):
            data: dict[A, list[B]]
        
        # Test annotation extraction with concrete types
        info = get_generic_info(NestedContainer[str, int])
        assert info.origin is NestedContainer
        assert info.type_params == []  # No TypeVars in current annotation (str, int are concrete)
        assert info.concrete_args == [str, int]
        
        # Test annotation extraction with TypeVars
        info = get_generic_info(NestedContainer[A, B])
        assert info.origin is NestedContainer
        assert set(info.type_params) == {A, B}  # TypeVars present in current annotation
        assert info.concrete_args == [A, B]
        
        # Test instance extraction  
        instance = NestedContainer[str, int](data={"key": [1, 2, 3]})
        instance.__orig_class__ = NestedContainer[str, int]  # Simulate Python behavior
        
        info = get_instance_generic_info(instance)
        assert info.origin is NestedContainer
        assert info.concrete_args == [str, int]


class TestUnionExtractor:
    """Test Union type extraction."""
    
    def test_union_annotation(self):
        extractor = UnionExtractor()
        
        # Simple Union
        assert extractor.can_handle_annotation(Union[int, str])
        
        info = extractor.extract_from_annotation(Union[int, str])
        assert info.origin is Union
        assert info.concrete_args == [int, str]
        assert info.type_params == []
        assert info.is_generic
        
        # Union with TypeVars
        info = extractor.extract_from_annotation(Union[A, int])
        assert info.origin is Union
        assert info.concrete_args == [A, int]
        assert info.type_params == [A]
        assert info.is_generic
        
        # Nested Union
        info = extractor.extract_from_annotation(Union[list[A], dict[B, int]])
        assert info.origin is Union
        assert set(info.type_params) == {A, B}
        assert info.concrete_args == [list[A], dict[B, int]]
    
    @pytest.mark.skipif(not hasattr(types, 'UnionType'), reason="types.UnionType not available")
    def test_modern_union_syntax(self):
        """Test modern union syntax (int | str)."""
        extractor = UnionExtractor()
        
        # Modern union syntax
        modern_union = int | str
        assert extractor.can_handle_annotation(modern_union)
        
        info = extractor.extract_from_annotation(modern_union)
        assert info.origin is getattr(types, 'UnionType')
        assert set(info.concrete_args) == {int, str}
        assert info.is_generic
    
    def test_union_instances(self):
        """Union types don't have direct instances."""
        extractor = UnionExtractor()
        
        # Union types can't be instantiated
        assert not extractor.can_handle_instance("hello")
        assert not extractor.can_handle_instance(42)
        
        # Should return simple type info for instances
        info = extractor.extract_from_instance("hello")
        assert info.origin is str
        assert not info.is_generic


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_union_if_needed(self):
        """Test union creation utility."""
        
        # Single type - should return the type itself
        result = create_union_if_needed({int})
        assert result is int
        
        # Multiple types - should create union
        result = create_union_if_needed({int, str})
        assert _is_union_type(result)
        assert set(get_args(result)) == {int, str}
        
        # Empty set - should return NoneType
        result = create_union_if_needed(set())
        assert result is type(None)
        
        # Test with complex types
        result = create_union_if_needed({list[int], dict[str, int]})
        assert _is_union_type(result)
        union_args = get_args(result)
        assert list[int] in union_args
        assert dict[str, int] in union_args


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"]) 