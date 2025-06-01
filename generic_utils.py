"""
Unified utilities for extracting type information from generic annotations and instances.

This module provides a consistent interface for working with generic types across
different systems (built-in generics, Pydantic models, dataclasses) without
needing special case handling in the main type inference code.

Key concepts:
- type_params: TypeVars that are present in the current annotation being analyzed
- concrete_args: The actual type arguments passed to the generic type
- origin: The base generic type (e.g., list for list[int])

For example:
- list[A] -> type_params=[A], concrete_args=[A], origin=list
- list[int] -> type_params=[], concrete_args=[int], origin=list
- DataclassBox[str] -> type_params=[], concrete_args=[str], origin=DataclassBox
"""

import typing
import types
from typing import Any, List, Tuple, TypeVar, get_origin, get_args, Union
from dataclasses import is_dataclass
from abc import ABC, abstractmethod


class GenericInfo:
    """
    Container for generic type information extracted from annotations or instances.
    
    Attributes:
        origin: The base generic type (e.g., list for list[int], DataclassBox for DataclassBox[str])
        type_params: TypeVars that are present in the current annotation (not the original class definition)
        concrete_args: The actual type arguments passed to the generic type
        is_generic: Whether this type has generic information (type_params or concrete_args)
    
    Examples:
        For list[A]: origin=list, type_params=[A], concrete_args=[A], is_generic=True
        For list[int]: origin=list, type_params=[], concrete_args=[int], is_generic=True
        For str: origin=str, type_params=[], concrete_args=[], is_generic=False
    """
    
    def __init__(
        self, 
        origin: Any = None, 
        type_params: List[TypeVar] = None, 
        concrete_args: List[Any] = None,
        is_generic: bool = False
    ):
        self.origin = origin
        self.type_params = type_params or []
        self.concrete_args = concrete_args or []
        self.is_generic = is_generic
    
    def __repr__(self):
        return (f"GenericInfo(origin={self.origin}, "
                f"type_params={self.type_params}, "
                f"concrete_args={self.concrete_args}, "
                f"is_generic={self.is_generic})")


class GenericExtractor(ABC):
    """Abstract base for type-system-specific generic extractors."""
    
    @abstractmethod
    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation."""
        pass
    
    @abstractmethod
    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance."""
        pass
    
    @abstractmethod
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a type annotation."""
        pass
    
    @abstractmethod
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from an instance."""
        pass


class BuiltinExtractor(GenericExtractor):
    """Extractor for built-in generic types like list, dict, tuple, set."""
    
    _builtin_origins = {list, dict, tuple, set, 
                       typing.List, typing.Dict, typing.Tuple, typing.Set}
    
    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in self._builtin_origins
    
    def can_handle_instance(self, instance: Any) -> bool:
        return type(instance) in {list, dict, tuple, set}
    
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        # For built-ins, we need to extract TypeVars from nested structures
        type_params = []
        for arg in args:
            if isinstance(arg, TypeVar):
                type_params.append(arg)
            else:
                # Recursively extract TypeVars from nested structures
                nested_params = extract_all_typevars(arg)
                type_params.extend(nested_params)
        
        concrete_args = list(args)
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=concrete_args,
            is_generic=bool(args)
        )
    
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        origin = type(instance)
        
        # Built-in instances don't preserve generic type info
        # We can only infer from content
        concrete_args = self._infer_args_from_content(instance)
        
        return GenericInfo(
            origin=origin,
            type_params=[],  # No TypeVar info available from instances
            concrete_args=concrete_args,
            is_generic=bool(concrete_args)
        )
    
    def _infer_args_from_content(self, instance: Any) -> List[Any]:
        """Infer type arguments from instance content."""
        if isinstance(instance, list) and instance:
            element_types = {self._infer_type_from_value(item) for item in instance}
            return [create_union_if_needed(element_types)]
        elif isinstance(instance, dict) and instance:
            key_types = {self._infer_type_from_value(k) for k in instance.keys()}
            value_types = {self._infer_type_from_value(v) for v in instance.values()}
            return [create_union_if_needed(key_types), create_union_if_needed(value_types)]
        elif isinstance(instance, tuple) and instance:
            return [self._infer_type_from_value(item) for item in instance]
        elif isinstance(instance, set) and instance:
            element_types = {self._infer_type_from_value(item) for item in instance}
            return [create_union_if_needed(element_types)]
        return []
    
    def _infer_type_from_value(self, value: Any) -> type:
        """Infer the most specific type from a value, including nested generic types."""
        info = get_instance_generic_info(value)
        if info.is_generic:
            return info.origin[*info.concrete_args]
        else:
            return info.origin


class PydanticExtractor(GenericExtractor):
    """Extractor for Pydantic generic models."""
    
    def can_handle_annotation(self, annotation: Any) -> bool:
        return hasattr(annotation, '__pydantic_generic_metadata__')
    
    def can_handle_instance(self, instance: Any) -> bool:
        return hasattr(instance, '__pydantic_generic_metadata__')
    
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        if not hasattr(annotation, '__pydantic_generic_metadata__'):
            return GenericInfo()
        
        metadata = annotation.__pydantic_generic_metadata__
        
        # Check if this is a specialized annotation (has 'origin' pointing to unparameterized base)
        if metadata.get('origin'):
            # This is a specialized annotation (e.g., PydanticBox[int])
            # Extract both origin and concrete args directly from the specialized annotation
            origin = metadata['origin']  # The unparameterized base class
            concrete_args = list(metadata.get('args', ()))
            
            # Get TypeVars from the origin class
            if hasattr(origin, '__pydantic_generic_metadata__'):
                origin_metadata = origin.__pydantic_generic_metadata__
                type_params = [p for p in origin_metadata.get('parameters', ()) if isinstance(p, TypeVar)]
            else:
                type_params = []
        else:
            # This is already the unparameterized base class (e.g., PydanticBox)
            origin = annotation
            type_params = [p for p in metadata.get('parameters', ()) if isinstance(p, TypeVar)]
            concrete_args = []  # No concrete args for unparameterized annotations
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=concrete_args,
            is_generic=bool(type_params or concrete_args)
        )
    
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not hasattr(instance, '__pydantic_generic_metadata__'):
            return GenericInfo()
        
        instance_class = type(instance)
        metadata = instance_class.__pydantic_generic_metadata__
        
        # Check if this is a specialized class (has 'origin' pointing to unparameterized base)
        if metadata.get('origin'):
            # This is a specialized class (e.g., PydanticBox[int])
            # Extract both origin and concrete args directly from the specialized class
            origin = metadata['origin']  # The unparameterized base class
            concrete_args = list(metadata.get('args', ()))
            
            # Get TypeVars from the origin class
            if hasattr(origin, '__pydantic_generic_metadata__'):
                origin_metadata = origin.__pydantic_generic_metadata__
                type_params = [p for p in origin_metadata.get('parameters', ()) if isinstance(p, TypeVar)]
            else:
                type_params = []
        else:
            # This is already the unparameterized base class (e.g., PydanticBox)
            origin = instance_class
            type_params = [p for p in metadata.get('parameters', ()) if isinstance(p, TypeVar)]
            concrete_args = []  # No concrete args for unparameterized instances
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=concrete_args,
            is_generic=bool(type_params or concrete_args)
        )


class UnionExtractor(GenericExtractor):
    """Extractor for Union types (both typing.Union and types.UnionType)."""
    
    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        union_type = getattr(types, 'UnionType', None)
        return origin is Union or (union_type and origin is union_type)
    
    def can_handle_instance(self, instance: Any) -> bool:
        # Instances don't have Union types directly
        return False
    
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract Union type information."""
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        # Extract TypeVars from the union arguments
        type_params = []
        for arg in args:
            if isinstance(arg, TypeVar):
                type_params.append(arg)
            else:
                # Recursively extract TypeVars from nested structures
                nested_params = extract_all_typevars(arg)
                type_params.extend(nested_params)
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=list(args),
            is_generic=bool(type_params or args)
        )
    
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Union types don't have instances directly."""
        return GenericInfo(origin=type(instance), is_generic=False)


class DataclassExtractor(GenericExtractor):
    """Extractor for dataclass generic types."""
    
    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation) or annotation
        return is_dataclass(origin) and hasattr(origin, '__orig_bases__')
    
    def can_handle_instance(self, instance: Any) -> bool:
        return is_dataclass(instance)
    
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        origin = get_origin(annotation) or annotation
        args = get_args(annotation)
        
        if not is_dataclass(origin):
            return GenericInfo()
        
        # For dataclasses, TypeVars are preserved directly in annotation args
        type_params = [arg for arg in args if isinstance(arg, TypeVar)]
        concrete_args = list(args)
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=concrete_args,
            is_generic=bool(args)
        )
    
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not is_dataclass(instance):
            return GenericInfo()
        
        origin = type(instance)
        
        # Check for __orig_class__ which preserves concrete type info
        concrete_args = []
        if hasattr(instance, '__orig_class__'):
            concrete_args = list(get_args(instance.__orig_class__))
        
        # Get TypeVar parameters from the class if it's generic
        type_params = []
        if hasattr(origin, '__orig_bases__'):
            for base in origin.__orig_bases__:
                base_args = get_args(base)
                type_params.extend(arg for arg in base_args if isinstance(arg, TypeVar))
        
        return GenericInfo(
            origin=origin,
            type_params=type_params,
            concrete_args=concrete_args,
            is_generic=bool(type_params or concrete_args)
        )


class GenericTypeUtils:
    """Unified interface for extracting generic type information."""
    
    def __init__(self):
        self.extractors = [
            BuiltinExtractor(),
            PydanticExtractor(),
            DataclassExtractor(),
            UnionExtractor()
        ]
    
    def get_generic_info(self, annotation: Any) -> GenericInfo:
        """
        Extract generic type information from an annotation.
        
        This replaces the need for separate get_origin/get_args calls
        and handles all generic type systems uniformly.
        """
        for extractor in self.extractors:
            if extractor.can_handle_annotation(annotation):
                return extractor.extract_from_annotation(annotation)
        
        # Fallback for non-generic types
        return GenericInfo(origin=annotation, is_generic=False)
    
    def get_instance_generic_info(self, instance: Any) -> GenericInfo:
        """
        Extract generic type information from an instance.
        
        This handles the differences between how Pydantic models,
        dataclasses, and built-in types store concrete type information.
        """
        for extractor in self.extractors:
            if extractor.can_handle_instance(instance):
                return extractor.extract_from_instance(instance)
        
        # Fallback for non-generic instances
        return GenericInfo(origin=type(instance), is_generic=False)
    
    def get_type_parameters(self, annotation: Any) -> List[TypeVar]:
        """
        Extract TypeVar parameters from any generic annotation.
        
        This returns TypeVars that are present in the current annotation,
        not the original class definition. For example:
        - list[A] -> [A]
        - list[int] -> []
        - DataclassBox[A, B] -> [A, B]
        - DataclassBox[str, int] -> []
        
        This replaces custom TypeVar extraction logic.
        """
        info = self.get_generic_info(annotation)
        return info.type_params
    
    def get_concrete_args(self, annotation: Any) -> List[Any]:
        """
        Extract concrete type arguments from any generic annotation.
        
        This returns the actual type arguments passed to the generic type:
        - list[int] -> [int]
        - dict[str, A] -> [str, A]  
        - DataclassBox[str] -> [str]
        - list (unparameterized) -> []
        
        This replaces custom argument extraction logic.
        """
        info = self.get_generic_info(annotation)
        return info.concrete_args
    
    def get_instance_concrete_args(self, instance: Any) -> List[Any]:
        """
        Extract concrete type arguments from any generic instance.
        
        This handles the different ways generic instances store type info:
        - Built-in containers: infer from content ([1, 2] -> [int])
        - Pydantic models: extract from __pydantic_generic_metadata__
        - Dataclasses: extract from __orig_class__ if available
        
        This handles the different ways generic instances store type info.
        """
        info = self.get_instance_generic_info(instance)
        return info.concrete_args
    
    def get_generic_origin(self, annotation: Any) -> Any:
        """
        Get the generic origin from any annotation.
        
        This replaces get_origin with support for custom generic types.
        """
        info = self.get_generic_info(annotation)
        return info.origin
    
    def is_generic_type(self, annotation: Any) -> bool:
        """Check if an annotation represents a generic type."""
        info = self.get_generic_info(annotation)
        return info.is_generic
    
    def extract_all_typevars(self, annotation: Any) -> List[TypeVar]:
        """
        Recursively extract all TypeVars from a nested annotation structure.
        
        This replaces _extract_typevars_from_annotation.
        """
        typevars = []
        seen = set()  # Avoid duplicates
        
        self._extract_typevars_recursive(annotation, typevars, seen)
        return typevars
    
    def _extract_typevars_recursive(self, annotation: Any, typevars: List[TypeVar], seen: set):
        """Recursively extract TypeVars avoiding duplicates."""
        if isinstance(annotation, TypeVar):
            if annotation not in seen:
                typevars.append(annotation)
                seen.add(annotation)
            return
        
        # Handle Union types specially (both typing.Union and types.UnionType)
        origin = get_origin(annotation)
        if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
            args = get_args(annotation)
            for arg in args:
                self._extract_typevars_recursive(arg, typevars, seen)
            return
        
        info = self.get_generic_info(annotation)
        
        # Add TypeVars from this level
        for param in info.type_params:
            if param not in seen:
                typevars.append(param)
                seen.add(param)
        
        # Recursively process concrete args
        for arg in info.concrete_args:
            self._extract_typevars_recursive(arg, typevars, seen)


def create_union_if_needed(types_set: set) -> Any:
    """Create a Union type if needed, or return single type."""
    if len(types_set) == 1:
        return list(types_set)[0]
    elif len(types_set) > 1:
        try:
            # Try modern union syntax
            result = types_set.pop()
            for elem_type in types_set:
                result = result | elem_type
            return result
        except TypeError:
            # Fallback to typing.Union
            return Union[tuple(types_set)]
    else:
        return type(None)


# Global instance for convenience
generic_utils = GenericTypeUtils()

# Convenience functions that mirror the class methods
def get_generic_info(annotation: Any) -> GenericInfo:
    """Extract generic type information from an annotation."""
    return generic_utils.get_generic_info(annotation)


def get_instance_generic_info(instance: Any) -> GenericInfo:
    """Extract generic type information from an instance."""
    return generic_utils.get_instance_generic_info(instance)


def get_type_parameters(annotation: Any) -> List[TypeVar]:
    """Extract TypeVar parameters from any generic annotation."""
    return generic_utils.get_type_parameters(annotation)


def get_concrete_args(annotation: Any) -> List[Any]:
    """Extract concrete type arguments from any generic annotation."""
    return generic_utils.get_concrete_args(annotation)


def get_instance_concrete_args(instance: Any) -> List[Any]:
    """Extract concrete type arguments from any generic instance."""
    return generic_utils.get_instance_concrete_args(instance)


def get_generic_origin(annotation: Any) -> Any:
    """Get the generic origin from any annotation."""
    return generic_utils.get_generic_origin(annotation)


def is_generic_type(annotation: Any) -> bool:
    """Check if an annotation represents a generic type."""
    return generic_utils.is_generic_type(annotation)


def extract_all_typevars(annotation: Any) -> List[TypeVar]:
    """Recursively extract all TypeVars from a nested annotation structure."""
    return generic_utils.extract_all_typevars(annotation) 