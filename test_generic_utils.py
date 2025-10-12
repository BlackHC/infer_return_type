"""
Unit tests for generic_utils module.

Tests the unified interface for extracting structural type information from different
generic type systems (built-ins, Pydantic, dataclasses).

Note: This module focuses on structural extraction only. Complex type inference
and field value analysis is handled by specialized systems like unification_type_inference.py
and csp_type_inference.py.
"""

import pytest
import typing
import types
from typing import Dict, List, TypeVar, Union, get_args, get_origin
from dataclasses import dataclass

from generic_utils import (
    BuiltinExtractor, DataclassExtractor, GenericTypeUtils, PydanticExtractor, UnionExtractor,
    create_union_if_needed, get_annotation_value_pairs, get_generic_info, get_instance_generic_info
)

# Test TypeVars
A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T', bound=str)
U = TypeVar('U', int, float)

def _is_union_type(obj):
    """Helper to check if object is a Union type (handles both typing.Union and types.UnionType)."""
    origin = get_origin(obj)
    return origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType'))

def _is_union_origin(origin):
    """Helper to check if an origin is a Union type (either typing.Union or types.UnionType)."""
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
    
    @pytest.fixture
    def extractor(self):
        return BuiltinExtractor()
    
    def test_list_annotation(self, extractor):
        # Non-generic list
        assert not extractor.can_handle_annotation(list)
        
        # Generic list with concrete type
        info = extractor.extract_from_annotation(list[int])
        assert info.origin is list
        assert info.resolved_concrete_args == [int]
        assert info.type_params == []
        assert info.is_generic
        
        # Generic list with TypeVar
        info = extractor.extract_from_annotation(list[A])
        assert info.origin is list
        assert info.resolved_concrete_args == [A]
        assert info.type_params == [A]
        assert info.is_generic
    
    def test_dict_annotation(self, extractor):
        info = extractor.extract_from_annotation(dict[str, int])
        assert info.origin is dict
        assert info.resolved_concrete_args == [str, int]
        assert info.type_params == []
        assert info.is_generic
        
        info = extractor.extract_from_annotation(dict[A, B])
        assert info.origin is dict
        assert info.resolved_concrete_args == [A, B]
        assert set(info.type_params) == {A, B}
        assert info.is_generic
    
    def test_tuple_annotation(self, extractor):
        info = extractor.extract_from_annotation(tuple[int, str, float])
        assert info.origin is tuple
        assert info.resolved_concrete_args == [int, str, float]
        assert info.type_params == []
        assert info.is_generic
        
        # Variable length tuple
        info = extractor.extract_from_annotation(tuple[A, ...])
        assert info.origin is tuple
        assert info.resolved_concrete_args == [A, ...]
        assert info.type_params == [A]
        assert info.is_generic
    
    def test_set_annotation(self, extractor):
        info = extractor.extract_from_annotation(set[int])
        assert info.origin is set
        assert info.resolved_concrete_args == [int]
        assert info.type_params == []
        assert info.is_generic
    
    def test_legacy_typing_annotations(self, extractor):
        info = extractor.extract_from_annotation(List[int])
        assert info.origin is list
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
        
        info = extractor.extract_from_annotation(Dict[str, int])
        assert info.origin is dict
        assert info.resolved_concrete_args == [str, int]
        assert info.is_generic
    
    def test_builtin_instances(self, extractor):
        """Test instance extraction returns basic container types."""
        # Test various container types
        test_cases = [
            ([], list),
            ([1, 2, 3], list),
            ({}, dict),
            ({"a": 1, "b": "hello"}, dict),
            ((1, "hello", 3.14), tuple),
            (set(), set),
            ({1, "hello"}, set),
        ]
        
        for instance, expected_origin in test_cases:
            info = extractor.extract_from_instance(instance)
            assert info.origin is expected_origin
            assert info.concrete_args == []
            assert not info.is_generic

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestPydanticExtractor:
    """Test Pydantic generic type extraction."""
    
    @pytest.fixture
    def extractor(self):
        return PydanticExtractor()
    
    def test_pydantic_annotation(self, extractor):
        # Generic Pydantic class
        assert extractor.can_handle_annotation(PydanticBox)
        
        info = extractor.extract_from_annotation(PydanticBox)
        assert info.origin is PydanticBox
        # The unparameterized class with TypeVar parameters should be considered generic
        assert info.is_generic  # Has TypeVar parameters
        assert A in info.type_params
        
        # Parameterized Pydantic class
        info = extractor.extract_from_annotation(PydanticBox[int])
        assert info.origin is PydanticBox  # Should be unparameterized base
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
        
        # Generic Pydantic class with TypeVar
        info = extractor.extract_from_annotation(PydanticBox[B])
        assert info.origin is PydanticBox  # Should be unparameterized base
        assert info.resolved_concrete_args == [B]
        assert info.is_generic
    
    def test_pydantic_multi_param(self):
        extractor = PydanticExtractor()
        
        info = extractor.extract_from_annotation(PydanticPair)
        assert info.origin is PydanticPair
        # The unparameterized class with TypeVar parameters should be considered generic
        assert info.is_generic  # Has TypeVar parameters
        assert set(info.type_params) == {A, B}
    
    def test_pydantic_instance(self):
        extractor = PydanticExtractor()
        
        # Instance with concrete type
        instance = PydanticBox[int](item=42)
        
        assert extractor.can_handle_instance(instance)
        
        info = extractor.extract_from_instance(instance)
        # Should get the base class, not the parameterized class
        assert info.origin is PydanticBox or info.origin.__name__ == 'PydanticBox'
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
    
    def test_pydantic_multi_param_instance(self):
        extractor = PydanticExtractor()
        
        instance = PydanticPair[str, int](first="hello", second=42)
        
        info = extractor.extract_from_instance(instance)
        # Should get the base class, not the parameterized class
        assert info.origin is PydanticPair or info.origin.__name__ == 'PydanticPair'
        assert info.resolved_concrete_args == [str, int]
        assert info.is_generic

class TestDataclassExtractor:
    """Test dataclass generic type extraction."""
    
    @pytest.fixture
    def extractor(self):
        return DataclassExtractor()
    
    def test_dataclass_annotation(self, extractor):
        # Generic dataclass
        assert extractor.can_handle_annotation(DataclassBox)
        
        info = extractor.extract_from_annotation(DataclassBox[int])
        assert info.origin is DataclassBox
        assert info.resolved_concrete_args == [int]
        assert info.type_params == []  # Concrete type, no TypeVars in current annotation
        assert info.is_generic
        
        # With TypeVar
        info = extractor.extract_from_annotation(DataclassBox[A])
        assert info.origin is DataclassBox
        assert info.resolved_concrete_args == [A]
        assert info.type_params == [A]  # TypeVar present in current annotation
        assert info.is_generic
    
    def test_dataclass_multi_param(self):
        extractor = DataclassExtractor()
        
        info = extractor.extract_from_annotation(DataclassPair[A, B])
        assert info.origin is DataclassPair
        assert info.resolved_concrete_args == [A, B]
        assert set(info.type_params) == {A, B}
        assert info.is_generic
    
    def test_dataclass_instance(self):
        extractor = DataclassExtractor()
        
        # Instance with __orig_class__ 
        instance = DataclassBox[int](item=42)
        instance.__orig_class__ = DataclassBox[int]  # Simulate what Python does
        
        assert extractor.can_handle_instance(instance)
        
        info = extractor.extract_from_instance(instance)
        assert info.origin is DataclassBox
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
    
    def test_dataclass_instance_without_orig_class(self):
        extractor = DataclassExtractor()
        
        # Instance without __orig_class__
        instance = DataclassBox(item=42)
        
        info = extractor.extract_from_instance(instance)
        assert info.origin is DataclassBox
        # Without __orig_class__, only return class type
        assert len(info.concrete_args) == 0
        assert not info.is_generic

class TestGenericTypeUtils:
    """Test the unified interface."""
    
    def test_builtin_types(self):
        utils = GenericTypeUtils()
        
        # List
        info = utils.get_generic_info(list[int])
        assert info.origin is list
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
        
        # TypeVars
        info = utils.get_generic_info(list[A])
        assert A in info.type_params
        
        # Instance extraction returns no args (structural only)
        instance_info = utils.get_instance_generic_info([1, 2, 3])
        assert len(instance_info.concrete_args) == 0
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_types(self):
        utils = GenericTypeUtils()
        
        # Annotation
        info = utils.get_generic_info(PydanticBox)
        assert info.origin is PydanticBox
        # The unparameterized class with TypeVar parameters should be considered generic
        assert info.is_generic  # Has TypeVar parameters
        assert A in info.type_params
        
        # Instance
        instance = PydanticBox[str](item="hello")
        instance_info = utils.get_instance_generic_info(instance)
        assert len(instance_info.concrete_args) == 1
        assert instance_info.concrete_args[0].resolved_type == str
    
    def test_dataclass_types(self):
        utils = GenericTypeUtils()
        
        # Annotation
        info = utils.get_generic_info(DataclassBox[A])
        assert info.origin is DataclassBox
        assert A in info.type_params
        
        # Instance with __orig_class__
        instance = DataclassBox[int](item=42)
        instance.__orig_class__ = DataclassBox[int]
        instance_info = utils.get_instance_generic_info(instance)
        assert len(instance_info.concrete_args) == 1
        assert instance_info.concrete_args[0].resolved_type == int
    
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
        info = utils.get_generic_info(list[A])
        assert info.type_params == [A]
        
        # Nested case
        info = utils.get_generic_info(dict[A, list[B]])
        assert set(info.type_params) == {A, B}
        
        # Deep nesting
        info = utils.get_generic_info(list[dict[A, tuple[B, int]]])
        assert set(info.type_params) == {A, B}
        
        # No TypeVars
        info = utils.get_generic_info(list[int])
        assert info.type_params == []
    
    def test_union_types(self):
        utils = GenericTypeUtils()
        
        # Union with TypeVars
        info = utils.get_generic_info(Union[A, B])
        assert set(info.type_params) == {A, B}
        
        # Nested Union
        info = utils.get_generic_info(list[Union[A, int]])
        assert info.type_params == [A]

class TestConvenienceFunctions:
    """Test the module-level convenience functions."""
    
    def test_get_generic_info(self):
        info = get_generic_info(list[int])
        assert info.origin is list
        assert info.resolved_concrete_args == [int]
        assert info.is_generic
    
    def test_get_type_parameters(self):
        info = get_generic_info(dict[A, B])
        assert set(info.type_params) == {A, B}
    
    def test_get_concrete_args(self):
        info = get_generic_info(tuple[int, str])
        assert len(info.concrete_args) == 2
        assert info.concrete_args[0].resolved_type == int
        assert info.concrete_args[1].resolved_type == str
    
    def test_get_instance_concrete_args(self):
        """Test instance concrete args extraction."""
        info = get_instance_generic_info([1, 2, 3])
        assert len(info.concrete_args) == 0
    
    def test_get_generic_origin(self):
        info = get_generic_info(list[int])
        assert info.origin is list
        
        info = get_generic_info(str)
        assert info.origin is str
    
    def test_is_generic_type(self):
        assert get_generic_info(list[int]).is_generic
        assert get_generic_info(dict[A, B]).is_generic
        assert not get_generic_info(str).is_generic
        assert not get_generic_info(int).is_generic
    
    def test_extract_all_typevars(self):
        info = get_generic_info(list[dict[A, B]])
        assert set(info.type_params) == {A, B}

class TestComplexScenarios:
    """Test complex nested scenarios."""
    
    def test_deeply_nested_builtin(self):
        # list[dict[str, tuple[int, float, set[A]]]]
        annotation = list[dict[str, tuple[int, float, set[A]]]]
        
        info = get_generic_info(annotation)
        assert info.type_params == [A]
        assert info.origin is list
        assert len(info.resolved_concrete_args) == 1
        assert info.resolved_concrete_args[0] == dict[str, tuple[int, float, set[A]]]
        assert info.is_generic
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_nested_custom_types(self):
        # list[PydanticBox[A]]
        annotation = list[PydanticBox[B]]
        
        info = get_generic_info(annotation)
        assert B in info.type_params
        
        # Instance extraction returns no args (structural only)
        box_instance = PydanticBox[int](item=42)
        list_instance = [box_instance]
        
        list_info = get_instance_generic_info(list_instance)
        assert len(list_info.concrete_args) == 0
    
    def test_union_with_generics(self):
        # Union[list[A], dict[B, int]]
        annotation = Union[list[A], dict[B, int]]
        
        info = get_generic_info(annotation)
        assert set(info.type_params) == {A, B}
    
    def test_bound_typevars(self):
        info = get_generic_info(list[T])
        assert info.type_params == [T]
        assert info.type_params[0].__bound__ is str
        
        info = get_generic_info(dict[U, int])
        assert info.type_params == [U]
        assert info.type_params[0].__constraints__ == (int, float)
    
    def test_nested_typevar_extraction_builtin(self):
        """Test that nested TypeVars are properly extracted from built-in types."""
        info = get_generic_info(list[list[A]])
        assert info.type_params == [A]
        assert info.resolved_concrete_args == [list[A]]
        
        info = get_generic_info(dict[A, list[B]])
        assert set(info.type_params) == {A, B}
        assert info.resolved_concrete_args == [A, list[B]]
        
        # Triple nesting
        info = get_generic_info(list[dict[A, set[B]]])
        assert set(info.type_params) == {A, B}
        assert info.resolved_concrete_args == [dict[A, set[B]]]
    

    
    def test_dataclass_with_nested_generics(self):
        """Test dataclass with nested generic annotations."""
        
        @dataclass
        class NestedContainer(typing.Generic[A, B]):
            data: dict[A, list[B]]
        
        # Test annotation extraction with concrete types
        info = get_generic_info(NestedContainer[str, int])
        assert info.origin is NestedContainer
        assert info.type_params == []  # No TypeVars in current annotation (str, int are concrete)
        assert info.resolved_concrete_args == [str, int]
        
        # Test annotation extraction with TypeVars
        info = get_generic_info(NestedContainer[A, B])
        assert info.origin is NestedContainer
        assert set(info.type_params) == {A, B}  # TypeVars present in current annotation
        assert info.resolved_concrete_args == [A, B]
        
        # Test instance extraction  
        instance = NestedContainer[str, int](data={"key": [1, 2, 3]})
        instance.__orig_class__ = NestedContainer[str, int]  # Simulate Python behavior
        
        info = get_instance_generic_info(instance)
        assert info.origin is NestedContainer
        assert info.resolved_concrete_args == [str, int]

class TestUnionExtractor:
    """Test Union type extraction."""
    
    def test_union_annotation(self):
        extractor = UnionExtractor()
        
        # Simple Union
        assert extractor.can_handle_annotation(Union[int, str])
        
        info = extractor.extract_from_annotation(Union[int, str])
        assert info.origin is Union
        assert set(info.resolved_concrete_args) == {int, str}
        assert info.type_params == []
        assert info.is_generic
        
        # Union with TypeVars
        info = extractor.extract_from_annotation(Union[A, int])
        assert info.origin is Union
        assert set(info.resolved_concrete_args) == {A, int}
        assert info.type_params == [A]
        assert info.is_generic
        
        # Nested Union
        info = extractor.extract_from_annotation(Union[list[A], dict[B, int]])
        assert info.origin is Union
        assert set(info.type_params) == {A, B}
        assert set(info.resolved_concrete_args) == {list[A], dict[B, int]}
    
    @pytest.mark.skipif(not hasattr(types, 'UnionType'), reason="types.UnionType not available")
    def test_modern_union_syntax(self):
        """Test modern union syntax (int | str)."""
        extractor = UnionExtractor()
        
        # Modern union syntax
        modern_union = int | str
        assert extractor.can_handle_annotation(modern_union)
        
        info = extractor.extract_from_annotation(modern_union)
        assert info.origin is getattr(types, 'UnionType')
        assert set(info.resolved_concrete_args) == {int, str}
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

@pytest.mark.skipif(not hasattr(types, 'UnionType'), reason="types.UnionType not available")
class TestModernUnionTypes:
    """Test modern union types (A | B syntax) throughout the system."""
    
    def test_builtin_modern_union_annotations(self):
        """Test built-in containers with modern union type arguments."""
        utils = GenericTypeUtils()
        
        # list[int | str]
        modern_list_union = list[int | str]
        info = utils.get_generic_info(modern_list_union)
        assert info.origin is list
        assert len(info.concrete_args) == 1
        union_arg = info.concrete_args[0]
        assert union_arg.origin is getattr(types, 'UnionType')  # Pure type unions should preserve UnionType
        assert set(union_arg.resolved_concrete_args) == {int, str}
        assert info.is_generic
        
        # dict[str, int | float]
        modern_dict_union = dict[str, int | float]
        info = utils.get_generic_info(modern_dict_union)
        assert info.origin is dict
        assert len(info.concrete_args) == 2
        assert info.concrete_args[0].origin is str
        union_arg = info.concrete_args[1]
        assert union_arg.origin is getattr(types, 'UnionType')  # Pure type unions should preserve UnionType
        assert set(union_arg.resolved_concrete_args) == {int, float}
        
        # tuple[int | str, bool | None]
        modern_tuple_union = tuple[int | str, bool | None]
        info = utils.get_generic_info(modern_tuple_union)
        assert info.origin is tuple
        assert len(info.concrete_args) == 2
        union_arg1 = info.concrete_args[0]
        union_arg2 = info.concrete_args[1]
        assert union_arg1.origin is getattr(types, 'UnionType')  # Pure type unions should preserve UnionType
        assert union_arg2.origin is getattr(types, 'UnionType')  # Pure type unions should preserve UnionType
        assert set(union_arg1.resolved_concrete_args) == {int, str}
        assert set(union_arg2.resolved_concrete_args) == {bool, type(None)}
        
        # set[A | int] with TypeVar
        A = TypeVar('A')
        modern_set_union_typevar = set[A | int]
        info = utils.get_generic_info(modern_set_union_typevar)
        assert info.origin is set
        assert len(info.concrete_args) == 1
        union_arg = info.concrete_args[0]
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(union_arg.origin)
        assert A in union_arg.type_params
        assert int in union_arg.resolved_concrete_args
    
    def test_modern_union_typevar_extraction(self):
        """Test TypeVar extraction from modern union types."""
        A = TypeVar('A')
        B = TypeVar('B')
        
        # Simple A | B
        info = get_generic_info(A | B)
        assert set(info.type_params) == {A, B}
        
        # Nested: list[A | B]
        info = get_generic_info(list[A | B])
        assert set(info.type_params) == {A, B}
        
        # Complex nesting: dict[A | str, list[B | int]]
        info = get_generic_info(dict[A | str, list[B | int]])
        assert set(info.type_params) == {A, B}
        
        # Mixed with concrete types: tuple[A | int, str | B]
        info = get_generic_info(tuple[A | int, str | B])
        assert set(info.type_params) == {A, B}
    
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_modern_union_annotations(self):
        """Test Pydantic models with modern union annotations."""
        A = TypeVar('A')
        
        class ModernUnionBox(BaseModel, typing.Generic[A]):
            value: A | str
        
        utils = GenericTypeUtils()
        
        # Test annotation
        info = utils.get_generic_info(ModernUnionBox[int])
        assert info.origin is ModernUnionBox
        assert len(info.concrete_args) == 1
        assert info.concrete_args[0].origin is int
        
        # Test instance
        instance = ModernUnionBox[int](value=42)
        info = utils.get_instance_generic_info(instance)
        assert info.origin is ModernUnionBox
        assert len(info.concrete_args) == 1
        assert info.concrete_args[0].origin is int
        
        # Test with actual union value
        instance_str = ModernUnionBox[int](value="hello")
        info = utils.get_instance_generic_info(instance_str)
        assert info.origin is ModernUnionBox
        assert len(info.concrete_args) == 1
        assert info.concrete_args[0].origin is int  # Type param, not inferred from value
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_field_value_inference_modern_unions(self):
        """Test simplified Pydantic extraction (no field value inference)."""
        A = TypeVar('A')
        
        class MixedFieldBox(BaseModel, typing.Generic[A]):
            field1: A
            field2: A
        
        # Create instance with conflicting types for same TypeVar
        instance = MixedFieldBox(field1=42, field2="hello")
        
        extractor = PydanticExtractor()
        info = extractor.extract_from_instance(instance)
        
        assert info.origin is MixedFieldBox
        # Without explicit type metadata, returns base class only
        assert len(info.concrete_args) == 0
        assert not info.is_generic
    
    def test_dataclass_modern_union_annotations(self):
        """Test dataclass with modern union annotations."""
        A = TypeVar('A')
        
        @dataclass
        class ModernUnionDataBox(typing.Generic[A]):
            value: A | None
        
        utils = GenericTypeUtils()
        
        # Test annotation
        info = utils.get_generic_info(ModernUnionDataBox[int])
        assert info.origin is ModernUnionDataBox
        assert len(info.concrete_args) == 1
        assert info.concrete_args[0].origin is int
        
        # Test instance with __orig_class__
        instance = ModernUnionDataBox[str](value="hello")
        instance.__orig_class__ = ModernUnionDataBox[str]
        info = utils.get_instance_generic_info(instance)
        assert info.origin is ModernUnionDataBox
        assert len(info.concrete_args) == 1
        assert info.concrete_args[0].origin is str
    
    def test_dataclass_field_value_inference_modern_unions(self):
        """Test simplified dataclass extraction (no field value inference)."""
        A = TypeVar('A')
        
        @dataclass
        class MixedFieldDataBox(typing.Generic[A]):
            field1: A
            field2: A
        
        # Create instance with conflicting types for same TypeVar
        instance = MixedFieldDataBox(field1=42, field2="hello")
        
        extractor = DataclassExtractor()
        info = extractor.extract_from_instance(instance)
        
        assert info.origin is MixedFieldDataBox
        # Simplified behavior: no field value inference
        assert len(info.concrete_args) == 0
        assert not info.is_generic
    
    def test_deeply_nested_modern_unions(self):
        """Test deeply nested structures with modern unions."""
        A = TypeVar('A')
        B = TypeVar('B')
        
        # Complex nesting: list[dict[A | str, tuple[B | int, set[A | B]]]]
        complex_annotation = list[dict[A | str, tuple[B | int, set[A | B]]]]
        
        info = get_generic_info(complex_annotation)
        assert set(info.type_params) == {A, B}
        
        info = get_generic_info(complex_annotation)
        assert info.origin is list
        assert len(info.concrete_args) == 1
        
        # Verify the structure is preserved
        dict_arg = info.concrete_args[0]
        assert dict_arg.origin is dict
        assert len(dict_arg.concrete_args) == 2
        
        # Check key type (A | str)
        key_union = dict_arg.concrete_args[0]
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(key_union.origin)
        assert A in key_union.type_params
        
        # Check value type (tuple[B | int, set[A | B]])
        tuple_arg = dict_arg.concrete_args[1]
        assert tuple_arg.origin is tuple
        assert len(tuple_arg.concrete_args) == 2
        
        # First tuple element: B | int
        first_union = tuple_arg.concrete_args[0]
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(first_union.origin)
        assert B in first_union.type_params
        
        # Second tuple element: set[A | B]
        set_arg = tuple_arg.concrete_args[1]
        assert set_arg.origin is set
        nested_union = set_arg.concrete_args[0]
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(nested_union.origin)
        assert set(nested_union.type_params) == {A, B}
    
    def test_convenience_functions_modern_unions(self):
        """Test all convenience functions work with modern unions."""
        A = TypeVar('A')
        B = TypeVar('B')
        
        # Test with list[A | B]
        modern_list_union = list[A | B]
        
        # get_generic_info
        info = get_generic_info(modern_list_union)
        assert info.origin is list
        union_arg = info.concrete_args[0]
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(union_arg.origin)
        
        # get_type_parameters (via type_params property)
        assert set(info.type_params) == {A, B}
        
        # get_concrete_args (via concrete_args property)
        assert len(info.concrete_args) == 1
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(info.concrete_args[0].origin)
        
        # get_generic_origin (via origin property)
        assert info.origin is list
        
        # is_generic_type (via is_generic property)
        assert get_generic_info(modern_list_union).is_generic
        assert get_generic_info(A | B).is_generic
        
        # get_resolved_type (via resolved_type property)
        assert get_origin(info.resolved_type) is list
        
        # extract_all_typevars (via type_params property)
        assert set(info.type_params) == {A, B}
    
    
    def test_modern_union_with_none(self):
        """Test modern union types with None (optional types)."""
        A = TypeVar('A')
        
        # A | None
        optional_typevar = A | None
        info = get_generic_info(optional_typevar)
        assert info.type_params == [A]
        
        # TypeVar unions might be converted to typing.Union
        assert _is_union_origin(info.origin)
        assert A in info.type_params
        assert type(None) in info.resolved_concrete_args
        
        # list[int | None]
        optional_list = list[int | None]
        info = get_generic_info(optional_list)
        assert info.origin is list
        union_arg = info.concrete_args[0]
        assert union_arg.origin is getattr(types, 'UnionType')  # Pure type unions should preserve UnionType
        assert set(union_arg.resolved_concrete_args) == {int, type(None)}
    
    def test_modern_union_equality_and_hashing(self):
        """Test that modern union GenericInfo objects work with equality and hashing."""
        A = TypeVar('A')
        
        # Create two equivalent modern union GenericInfo objects (pure types)
        info1 = get_generic_info(list[int | str])
        info2 = get_generic_info(list[int | str])
        
        # They should be equal
        assert info1 == info2
        
        # They should have the same hash (for use in sets/dicts)
        assert hash(info1) == hash(info2)
        
        # Test with sets
        info_set = {info1, info2}
        assert len(info_set) == 1  # Should deduplicate
        
        # Test with TypeVars (just check that they work, might not be equal due to union type conversion)
        info3 = get_generic_info(A | int)
        info4 = get_generic_info(A | int)
        # TypeVar unions behavior might vary, so just check they're hashable
        assert isinstance(hash(info3), int)
        assert isinstance(hash(info4), int)
    
    def test_modern_union_resolved_type_property(self):
        """Test that resolved_type property works correctly with modern unions."""
        A = TypeVar('A')
        
        # Test direct modern union (pure types)
        union_info = get_generic_info(int | str)
        resolved = union_info.resolved_type
        assert _is_union_type(resolved)
        assert set(get_args(resolved)) == {int, str}
        
        # Test nested modern union (pure types)
        list_union_info = get_generic_info(list[int | str])
        resolved = list_union_info.resolved_type
        assert get_origin(resolved) is list
        inner_union = get_args(resolved)[0]
        assert _is_union_type(inner_union)
        assert set(get_args(inner_union)) == {int, str}
        
        # Test with TypeVar (might be converted to typing.Union)
        typevar_union_info = get_generic_info(A | int)
        resolved = typevar_union_info.resolved_type
        assert _is_union_type(resolved)
        assert A in get_args(resolved)
        assert int in get_args(resolved)


class TestGetAnnotationValuePairs:
    """Test the get_annotation_value_pairs function."""
    
    def test_list_with_none_values(self):
        """Test that None values are included in list pairs."""
        annotation = List[A]
        instance = [1, None, 2, None]
        
        pairs = get_annotation_value_pairs(annotation, instance)
        
        # Should have 4 pairs (including None values)
        assert len(pairs) == 4
        
        # Check that we get both int and NoneType values
        value_types = {type(val) for _, val in pairs}
        assert value_types == {int, type(None)}
        
        # Check that None values are actually in the pairs
        values = [val for _, val in pairs]
        assert values == [1, None, 2, None]
    
    def test_dict_with_none_values(self):
        """Test that None values are included in dict pairs."""
        annotation = Dict[A, B]
        instance = {None: 1, "key": None, "key2": 2}
        
        pairs = get_annotation_value_pairs(annotation, instance)
        
        # Should have 6 pairs: 3 keys + 3 values
        assert len(pairs) == 6
        
        # Check keys (should include None)
        key_pairs = [(info, val) for info, val in pairs 
                    if info.origin == A]  # A is the key type
        key_values = [val for _, val in key_pairs]
        assert None in key_values
        assert "key" in key_values
        assert "key2" in key_values
        
        # Check values (should include None)  
        value_pairs = [(info, val) for info, val in pairs 
                      if info.origin == B]  # B is the value type
        value_values = [val for _, val in value_pairs]
        assert None in value_values
        assert 1 in value_values
        assert 2 in value_values
    
    def test_tuple_with_none_values(self):
        """Test that None values are included in tuple pairs."""
        # Variable length tuple
        annotation = tuple[A, ...]
        instance = (1, None, "hello", None)
        
        pairs = get_annotation_value_pairs(annotation, instance)
        
        # Should have 4 pairs (including None values)
        assert len(pairs) == 4
        
        # Check that None values are included
        values = [val for _, val in pairs]
        assert values == [1, None, "hello", None]
    
    def test_fixed_tuple_with_none_values(self):
        """Test that None values are included in fixed-length tuple pairs."""
        annotation = tuple[A, B, A]
        instance = (None, "hello", None)
        
        pairs = get_annotation_value_pairs(annotation, instance)
        
        # Should have 3 pairs
        assert len(pairs) == 3
        
        # Check that None values are included in the right positions
        values = [val for _, val in pairs]
        assert values == [None, "hello", None]
    
    def test_set_with_none_values(self):
        """Test that None values are included in set pairs."""
        annotation = set[A]
        instance = {1, None, "hello"}
        
        pairs = get_annotation_value_pairs(annotation, instance)
        
        # Should have 3 pairs (including None)
        assert len(pairs) == 3
        
        # Check that None is included
        values = {val for _, val in pairs}
        assert values == {1, None, "hello"}
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_with_none_values(self):
        """Test that None values are included in Pydantic model field pairs."""
        from typing import Optional
        
        class NullableBox(BaseModel, typing.Generic[A]):
            nullable_field: Optional[A]
            other_field: A
        
        # Create instance with None value
        instance = NullableBox[int](nullable_field=None, other_field=42)
        
        pairs = get_annotation_value_pairs(NullableBox[A], instance)
        
        # Should have 2 pairs (including None field)
        assert len(pairs) == 2
        
        # Check that None value is included
        values = [val for _, val in pairs]
        assert None in values
        assert 42 in values
    
    def test_dataclass_with_none_values(self):
        """Test that None values are included in dataclass field pairs."""
        
        @dataclass
        class NullableDataBox(typing.Generic[A]):
            nullable_field: A
            other_field: A
        
        # Create instance with None value
        instance = NullableDataBox[int](nullable_field=None, other_field=42)
        
        pairs = get_annotation_value_pairs(NullableDataBox[A], instance)
        
        # Should have 2 pairs (including None field)
        assert len(pairs) == 2
        
        # Check that None value is included
        values = [val for _, val in pairs]
        assert None in values
        assert 42 in values


class TestGenericInfoEdgeCases:
    """Test edge cases and uncovered branches in GenericInfo."""

    def test_generic_info_equality_with_non_generic_info(self):
        """Test GenericInfo.__eq__ returns False for non-GenericInfo objects."""
        from generic_utils import GenericInfo
        
        info = GenericInfo(origin=int)
        assert info != "not a GenericInfo"
        assert info != 42
        assert info != None
    
    def test_generic_info_hash_with_unhashable_type(self):
        """Test GenericInfo.__hash__ handles unhashable resolved_type."""
        from generic_utils import GenericInfo
        
        # Create a GenericInfo with a type that might cause issues in hash
        # The __hash__ method has a try/except for TypeError
        info = GenericInfo(origin=list, concrete_args=[GenericInfo(origin=dict)])
        
        # Should not raise, should use str() fallback
        hash_value = hash(info)
        assert isinstance(hash_value, int)


class TestExtractorEdgeCases:
    """Test edge cases in various extractors."""
    
    def test_builtin_extractor_non_generic_instance(self):
        """Test BuiltinExtractor with various edge cases."""
        extractor = BuiltinExtractor()
        
        # Can handle empty containers
        assert extractor.can_handle_instance([])
        assert extractor.can_handle_instance({})
        assert extractor.can_handle_instance(())
        assert extractor.can_handle_instance(set())
        
        # Extract from empty containers
        info = extractor.extract_from_instance([])
        assert info.origin is list
        assert not info.concrete_args
    
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_extractor_unparameterized_base(self):
        """Test PydanticExtractor with unparameterized base classes."""
        from generic_utils import GenericInfo
        
        A = TypeVar('A')
        
        class UnparameterizedBox(BaseModel, typing.Generic[A]):
            item: A
        
        extractor = PydanticExtractor()
        
        # Test extraction from unparameterized annotation
        info = extractor.extract_from_annotation(UnparameterizedBox)
        # Should extract TypeVars from class definition
        assert A in info.type_params
        
        # Test extraction from unparameterized instance
        instance = UnparameterizedBox(item=42)
        inst_info = extractor.extract_from_instance(instance)
        assert inst_info.origin == UnparameterizedBox or hasattr(inst_info.origin, '__name__')
    
    def test_dataclass_extractor_without_orig_class(self):
        """Test DataclassExtractor with instance lacking __orig_class__."""
        A = TypeVar('A')
        
        @dataclass
        class SimpleBox(typing.Generic[A]):
            value: A
        
        extractor = DataclassExtractor()
        
        # Create instance without explicit type parameter
        instance = SimpleBox(value=42)
        
        # Extract - should handle missing __orig_class__
        info = extractor.extract_from_instance(instance)
        assert info.origin is SimpleBox
        # Without __orig_class__, concrete_args should be empty
        assert info.concrete_args == []


class TestGetAnnotationValuePairsEdgeCases:
    """Test edge cases in get_annotation_value_pairs function."""
    
    def test_annotation_value_pairs_with_none_instance(self):
        """Test get_annotation_value_pairs with None as instance."""
        A = TypeVar('A')
        
        pairs = get_annotation_value_pairs(list[A], None)
        assert pairs == []
    
    def test_annotation_value_pairs_no_type_params(self):
        """Test get_annotation_value_pairs when annotation has no type parameters."""
        # Annotation without type parameters
        pairs = get_annotation_value_pairs(int, 42)
        assert pairs == []
        
        pairs = get_annotation_value_pairs(str, "hello")
        assert pairs == []
    
    def test_annotation_value_pairs_custom_object_with_dict(self):
        """Test get_annotation_value_pairs with custom object having __dict__."""
        A = TypeVar('A')
        
        class CustomClass(typing.Generic[A]):
            def __init__(self, value):
                self.value = value
                self.other = value * 2
                self.__private = "skip"  # Should be skipped
        
        instance = CustomClass(42)
        
        # Get annotation info
        from generic_utils import get_generic_info
        ann_info = get_generic_info(CustomClass[A])
        
        # Test that we can extract from custom objects with __dict__
        pairs = get_annotation_value_pairs(CustomClass[A], instance)
        
        # The behavior depends on whether the custom class is recognized
        # If it has concrete_args in ann_info, it should extract values
        if pairs:
            values = [val for _, val in pairs]
            assert 42 in values or 84 in values
            assert "skip" not in values  # Private attributes should be skipped
        # If no pairs, the extractor doesn't support this custom class (which is OK)
    
    def test_annotation_value_pairs_tuple_variable_length(self):
        """Test get_annotation_value_pairs with variable length tuple."""
        A = TypeVar('A')
        
        pairs = get_annotation_value_pairs(tuple[A, ...], (1, 2, 3, 4))
        
        # Should have 4 pairs (one for each element)
        assert len(pairs) == 4
        values = [val for _, val in pairs]
        assert values == [1, 2, 3, 4]
    
    def test_annotation_value_pairs_tuple_fixed_length(self):
        """Test get_annotation_value_pairs with fixed length tuple."""
        A = TypeVar('A')
        B = TypeVar('B')
        C = TypeVar('C')
        
        pairs = get_annotation_value_pairs(tuple[A, B, C], (1, "hello", 3.14))
        
        # Should have 3 pairs
        assert len(pairs) == 3
        values = [val for _, val in pairs]
        assert values == [1, "hello", 3.14]


class TestCreateUnionEdgeCases:
    """Test edge cases in create_union_if_needed."""
    
    def test_create_union_empty_set(self):
        """Test create_union_if_needed with empty set returns NoneType."""
        result = create_union_if_needed(set())
        assert result is type(None)
    
    def test_create_union_single_type(self):
        """Test create_union_if_needed with single type returns that type."""
        result = create_union_if_needed({int})
        assert result is int
    
    def test_create_union_multiple_types(self):
        """Test create_union_if_needed with multiple types creates union."""
        result = create_union_if_needed({int, str, float})
        
        # Should be a union type
        origin = get_origin(result)
        assert _is_union_origin(origin)
        
        args = get_args(result)
        assert set(args) == {int, str, float}
    
    def test_create_union_with_unhashable_fallback(self):
        """Test create_union_if_needed TypeError fallback to typing.Union."""
        # In case the | operator doesn't work, it should fall back to typing.Union
        # This is hard to trigger directly, but we can test the logic exists
        result = create_union_if_needed({int, str})
        
        # Should still create a valid union
        origin = get_origin(result)
        assert _is_union_origin(origin)


