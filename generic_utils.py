"""
Unified utilities for extracting type information from generic annotations and instances.

This module provides a consistent interface for working with generic types across
different systems (built-in generics, Pydantic models, dataclasses) without
needing special case handling in the main type inference code.

Key concepts:
- concrete_args: List[GenericInfo | TypeVar] - the actual type arguments passed to the generic type
- type_params: TypeVars computed from concrete_args and their nested type_params
- origin: The base generic type (e.g., list for list[int])
- resolved_type: The fully materialized type using origin[*resolved_args]

For example:
- list[A] -> concrete_args=[A], type_params=[A], origin=list
- list[int] -> concrete_args=[GenericInfo(int)], type_params=[], origin=list
- DataclassBox[str] -> concrete_args=[GenericInfo(str)], type_params=[], origin=DataclassBox
"""

import functools
import typing
import types
from typing import Any, List, Dict, Tuple, Set, TypeVar, get_origin, get_args, Union, Iterable
from dataclasses import is_dataclass, dataclass, field
from abc import ABC, abstractmethod


@dataclass(frozen=True, kw_only=True)
class GenericInfo:
    """
    Container for generic type information extracted from annotations or instances.

    Attributes:
        origin: The base generic type (e.g., list for list[int], DataclassBox for DataclassBox[str])
        concrete_args: List[GenericInfo] - the actual type arguments passed to the generic type, all wrapped as GenericInfo
        type_params: TypeVars computed from concrete_args and their nested type_params
        is_generic: Whether this type has generic information (type_params or concrete_args)
        resolved_type: The fully materialized type using origin[*resolved_args]

    Examples:
        For list[A]: origin=list, concrete_args=[GenericInfo(origin=A)], type_params=[A], is_generic=True
        For list[int]: origin=list, concrete_args=[GenericInfo(origin=int)], type_params=[], is_generic=True
        For str: origin=str, concrete_args=[], type_params=[], is_generic=False
    """

    origin: Any = None
    concrete_args: List["GenericInfo"] = field(default_factory=list)
    is_generic: bool = False
    type_params: List[TypeVar] = field(init=False)

    def __post_init__(self):
        # Compute type_params from concrete_args
        object.__setattr__(self, "type_params", self._compute_type_params())

    @staticmethod
    def make_union_if_needed(
        sub_generic_infos: Iterable["GenericInfo"],
    ) -> "GenericInfo":
        """Create a union type from a list of GenericInfo objects."""
        sub_generic_infos = set(sub_generic_infos)
        if len(sub_generic_infos) == 1:
            return next(iter(sub_generic_infos))
        elif len(sub_generic_infos) > 1:
            return GenericInfo(
                origin=Union, concrete_args=list(sub_generic_infos), is_generic=True
            )
        else:
            return GenericInfo(origin=type(None), is_generic=False)

    def _compute_type_params(self) -> List[TypeVar]:
        """Compute TypeVars from concrete_args and their nested type_params."""
        seen = set()  # Avoid duplicates

        for arg in self.concrete_args:
            if isinstance(arg.origin, TypeVar):
                seen.add(arg.origin)
            else:
                for nested_param in arg.type_params:
                    seen.add(nested_param)

        return list(seen)

    @functools.cached_property
    def resolved_type(self) -> Any:
        """The fully materialized type using origin[*resolved_args]."""
        if not self.concrete_args:
            return self.origin

        resolved_args = []
        for arg in self.concrete_args:
            resolved_args.append(arg.resolved_type)

        if get_origin(self.origin) is Union or (
            hasattr(types, "UnionType")
            and get_origin(self.origin) is getattr(types, "UnionType")
        ):
            # Handle Union types
            return create_union_if_needed(set(resolved_args))
        elif self.origin in (tuple, typing.Tuple):
            if len(resolved_args) == 2 and resolved_args[1] is ...:
                return tuple[resolved_args[0], ...]
            else:
                return tuple[tuple(resolved_args)]
        else:
            return self.origin[*resolved_args]
        
    @functools.cached_property
    def resolved_concrete_args(self) -> List[Any]:
        """The fully materialized concrete arguments using origin[*resolved_args]."""
        if not self.concrete_args:
            return self.concrete_args
        
        resolved_args = []
        for arg in self.concrete_args:
            resolved_args.append(arg.resolved_type)
        
        return resolved_args

    def __eq__(self, other):
        """Check equality based on origin and resolved_type."""
        if not isinstance(other, GenericInfo):
            return False
        return (
            self.origin == other.origin and 
            self.resolved_type == other.resolved_type
        )

    def __hash__(self):
        """Make GenericInfo hashable based on origin and resolved_type."""
        try:
            return hash((self.origin, self.resolved_type))
        except TypeError:
            # If resolved_type is not hashable, use origin and str representation
            return hash((self.origin, str(self.resolved_type)))


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

    _builtin_origins = {
        list,
        dict,
        tuple,
        set,
        List,
        Dict,
        Tuple,
        Set,
    }

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in self._builtin_origins

    def can_handle_instance(self, instance: Any) -> bool:
        return type(instance) in {list, dict, tuple, set}

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Convert args to GenericInfo
        concrete_args = []
        for arg in args:
            nested_info = get_generic_info(arg)
            concrete_args.append(nested_info)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(args)
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        origin = type(instance)

        # Built-in instances don't preserve generic type info
        # We can only infer from content
        concrete_args = self._infer_args_from_content(instance)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(concrete_args)
        )

    def _infer_args_from_content(self, instance: Any) -> List[GenericInfo]:
        """Infer type arguments from instance content."""
        if isinstance(instance, list) and instance:
            element_types = {get_instance_generic_info(item) for item in instance}
            return [GenericInfo.make_union_if_needed(element_types)]
        elif isinstance(instance, dict) and instance:
            key_types = {get_instance_generic_info(k) for k in instance.keys()}
            value_types = {get_instance_generic_info(v) for v in instance.values()}
            return [GenericInfo.make_union_if_needed(key_types), GenericInfo.make_union_if_needed(value_types)]
        elif isinstance(instance, tuple) and instance:
            return [
                get_instance_generic_info(item)
                for item in instance
            ]
        elif isinstance(instance, set) and instance:
            element_types = {get_instance_generic_info(item) for item in instance}
            return [GenericInfo.make_union_if_needed(element_types)]
        return []


class PydanticExtractor(GenericExtractor):
    """Extractor for Pydantic generic models."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        return hasattr(annotation, "__pydantic_generic_metadata__")

    def can_handle_instance(self, instance: Any) -> bool:
        return hasattr(instance, "__pydantic_generic_metadata__")

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        if not hasattr(annotation, "__pydantic_generic_metadata__"):
            return GenericInfo()

        metadata = annotation.__pydantic_generic_metadata__

        # Check if this is a specialized annotation (has 'origin' pointing to unparameterized base)
        if metadata.get("origin"):
            # This is a specialized annotation (e.g., PydanticBox[int])
            origin = metadata["origin"]  # The unparameterized base class
            args = metadata.get("args", ())

            # Convert args to GenericInfo
            concrete_args = []
            for arg in args:
                nested_info = get_generic_info(arg)
                concrete_args.append(nested_info)
        else:
            # This is the unparameterized base class (e.g., PydanticBox)
            origin = annotation
            concrete_args = []  # No concrete args for unparameterized annotations

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(concrete_args)
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not hasattr(instance, "__pydantic_generic_metadata__"):
            return GenericInfo()

        instance_class = type(instance)
        metadata = instance_class.__pydantic_generic_metadata__

        # Check if this is a specialized class (has 'origin' pointing to unparameterized base)
        if metadata.get("origin"):
            # This is a specialized class (e.g., PydanticBox[int])
            origin = metadata["origin"]  # The unparameterized base class
            args = metadata.get("args", ())

            # Convert args to GenericInfo
            concrete_args = []
            for arg in args:
                nested_info = get_generic_info(arg)
                concrete_args.append(nested_info)
        else:
            # This is the unparameterized base class (e.g., PydanticBox)
            origin = instance_class
            concrete_args = []  # No concrete args for unparameterized instances

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(concrete_args)
        )


class UnionExtractor(GenericExtractor):
    """Extractor for Union types (both typing.Union and types.UnionType)."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        union_type = getattr(types, "UnionType", None)
        return origin is Union or (union_type and origin is union_type)

    def can_handle_instance(self, instance: Any) -> bool:
        # Instances don't have Union types directly
        return False

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract Union type information."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Convert args to GenericInfo
        concrete_args = []
        for arg in args:
            nested_info = get_generic_info(arg)
            concrete_args.append(nested_info)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(concrete_args)
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Union types don't have instances directly."""
        return GenericInfo(origin=type(instance), is_generic=False)


class DataclassExtractor(GenericExtractor):
    """Extractor for dataclass generic types."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation) or annotation
        return is_dataclass(origin) and hasattr(origin, "__orig_bases__")

    def can_handle_instance(self, instance: Any) -> bool:
        return is_dataclass(instance)

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        origin = get_origin(annotation) or annotation
        args = get_args(annotation)

        if not is_dataclass(origin):
            return GenericInfo()

        # Convert args to GenericInfo
        concrete_args = []
        for arg in args:
            nested_info = get_generic_info(arg)
            concrete_args.append(nested_info)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(args)
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not is_dataclass(instance):
            return GenericInfo()

        origin = type(instance)

        # Check for __orig_class__ which preserves concrete type info
        concrete_args = []
        if hasattr(instance, "__orig_class__"):
            args = get_args(instance.__orig_class__)
            for arg in args:

                nested_info = get_generic_info(arg)
                concrete_args.append(nested_info)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args, is_generic=bool(concrete_args)
        )


class GenericTypeUtils:
    """Unified interface for extracting generic type information."""

    def __init__(self):
        self.extractors = [
            BuiltinExtractor(),
            PydanticExtractor(),
            DataclassExtractor(),
            UnionExtractor(),
        ]

    def get_generic_info(self, annotation: Any) -> GenericInfo:
        """
        Extract generic type information from an annotation.

        This replaces the need for separate get_origin/get_args calls
        and handles all generic type systems uniformly.
        """
        if isinstance(annotation, TypeVar):
            return GenericInfo(origin=annotation, is_generic=False)

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

        This returns TypeVars that are present in the current annotation.
        """
        info = self.get_generic_info(annotation)
        return info.type_params

    def get_concrete_args(self, annotation: Any) -> List[GenericInfo]:
        """
        Extract concrete type arguments from any generic annotation.

        This returns the actual type arguments passed to the generic type as GenericInfo objects.
        """
        info = self.get_generic_info(annotation)
        return info.concrete_args

    def get_instance_concrete_args(
        self, instance: Any
    ) -> List[GenericInfo]:
        """
        Extract concrete type arguments from any generic instance.

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

    def get_resolved_type(self, annotation: Any) -> Any:
        """
        Get the fully resolved/materialized type from an annotation.

        This returns the concrete type with all generics properly instantiated.
        """
        info = self.get_generic_info(annotation)
        return info.resolved_type

    def extract_all_typevars(self, annotation: Any) -> List[TypeVar]:
        """
        Recursively extract all TypeVars from a nested annotation structure.

        This replaces _extract_typevars_from_annotation.
        """
        generic_info = self.get_generic_info(annotation)
        return generic_info.type_params

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


def get_concrete_args(annotation: Any) -> List[GenericInfo]:
    """Extract concrete type arguments from any generic annotation."""
    return generic_utils.get_concrete_args(annotation)


def get_instance_concrete_args(instance: Any) -> List[GenericInfo]:
    """Extract concrete type arguments from any generic instance."""
    return generic_utils.get_instance_concrete_args(instance)


def get_generic_origin(annotation: Any) -> Any:
    """Get the generic origin from any annotation."""
    return generic_utils.get_generic_origin(annotation)


def is_generic_type(annotation: Any) -> bool:
    """Check if an annotation represents a generic type."""
    return generic_utils.is_generic_type(annotation)


def get_resolved_type(annotation: Any) -> Any:
    """Get the fully resolved/materialized type from an annotation."""
    return generic_utils.get_resolved_type(annotation)


def extract_all_typevars(annotation: Any) -> List[TypeVar]:
    """Recursively extract all TypeVars from a nested annotation structure."""
    return generic_utils.extract_all_typevars(annotation)
