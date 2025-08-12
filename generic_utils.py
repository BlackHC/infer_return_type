"""
Unified utilities for extracting type information from generic annotations and instances.

This module provides a consistent interface for working with generic types across
different systems (built-in generics, Pydantic models, dataclasses).

Key concepts:
- concrete_args: The actual type arguments as GenericInfo objects
- type_params: TypeVars extracted from the concrete arguments
- origin: The base generic type (e.g., list for list[int])
- resolved_type: The fully materialized type
"""

import functools
import typing
import types
from typing import Any, Dict, Iterable, List, Set, TypeVar, Tuple, Union, get_args, get_origin
from dataclasses import dataclass, field, fields, is_dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True, kw_only=True)
class GenericInfo:
    """Container for generic type information extracted from annotations or instances.

    Attributes:
        origin: The base generic type (e.g., list for list[int])
        concrete_args: The actual type arguments as GenericInfo objects
        is_generic: Whether this type has generic information
        type_params: TypeVars computed from concrete_args (derived field)
        resolved_type: The fully materialized type (cached property)
    """

    origin: Any = None
    concrete_args: List["GenericInfo"] = field(default_factory=list)
    type_params: List[TypeVar] = field(init=False)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        object.__setattr__(self, "type_params", self._compute_type_params())

    @property
    def is_generic(self) -> bool:
        """Whether this type has generic information (computed from concrete_args)."""
        return bool(self.concrete_args)

    @staticmethod
    def make_union_if_needed(sub_generic_infos: Iterable["GenericInfo"]) -> "GenericInfo":
        """Create a union type from a list of GenericInfo objects."""
        sub_generic_infos = set(sub_generic_infos)
        if len(sub_generic_infos) == 1:
            return next(iter(sub_generic_infos))
        elif len(sub_generic_infos) > 1:
            return GenericInfo(
                origin=Union, concrete_args=list(sub_generic_infos)
            )
        else:
            return GenericInfo(origin=type(None))

    def _compute_type_params(self) -> List[TypeVar]:
        """Compute TypeVars from concrete_args and their nested type_params."""
        seen = set()
        for arg in self.concrete_args:
            if isinstance(arg.origin, TypeVar):
                seen.add(arg.origin)
            else:
                seen.update(arg.type_params)
        return list(seen)

    @functools.cached_property
    def resolved_type(self) -> Any:
        """The fully materialized type using origin[*resolved_args]."""
        if not self.concrete_args:
            return self.origin

        resolved_args = [arg.resolved_type for arg in self.concrete_args]

        if self._is_union_origin():
            return create_union_if_needed(set(resolved_args))
        elif self.origin in (tuple, typing.Tuple):
            if len(resolved_args) == 2 and resolved_args[1] is ...:
                return tuple[resolved_args[0], ...]
            else:
                return tuple[tuple(resolved_args)]
        else:
            return self.origin[*resolved_args]

    def _is_union_origin(self) -> bool:
        """Check if origin is a Union type."""
        return (
            self.origin is Union 
            or (hasattr(types, "UnionType") and self.origin is getattr(types, "UnionType"))
        )
        
    @functools.cached_property
    def resolved_concrete_args(self) -> List[Any]:
        """The fully materialized concrete arguments."""
        return [arg.resolved_type for arg in self.concrete_args] if self.concrete_args else []

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
            return hash((self.origin, str(self.resolved_type)))


class GenericExtractor(ABC):
    """Abstract base for type-system-specific generic extractors."""

    @abstractmethod
    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation."""

    @abstractmethod
    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance."""

    @abstractmethod
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a type annotation."""

    @abstractmethod
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from an instance."""


class FieldBasedExtractor(GenericExtractor):
    """Base class for extractors that can infer types from field values."""

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a class definition."""
        for base in getattr(dataclass_class, "__orig_bases__", []):
            if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                origin = get_origin(base)
                if origin and hasattr(origin, "__name__") and "Generic" in str(origin):
                    args = get_args(base)
                    return [arg for arg in args if isinstance(arg, TypeVar)]
        return []

    def _map_typevars_to_inferred_types(
        self, type_params: List[TypeVar], field_annotations: Dict[str, Any], field_values: Dict[str, Any]
    ) -> List[GenericInfo]:
        """Map TypeVars to inferred types based on field values."""
        typevar_to_types: Dict[TypeVar, Set[Any]] = {tv: set() for tv in type_params}
        
        # Analyze each field
        for field_name, field_annotation in field_annotations.items():
            if field_name in field_values:
                field_value = field_values[field_name]
                self._extract_typevar_bindings(field_annotation, field_value, typevar_to_types)
        
        # Convert to GenericInfo objects in the same order as type_params
        concrete_args = []
        for type_param in type_params:
            inferred_types = typevar_to_types.get(type_param, set())
            if inferred_types:
                if len(inferred_types) == 1:
                    inferred_type = next(iter(inferred_types))
                    concrete_args.append(GenericInfo(origin=inferred_type))
                else:
                    # Multiple types, create union
                    union_info = GenericInfo.make_union_if_needed([
                        GenericInfo(origin=t) for t in inferred_types
                    ])
                    concrete_args.append(union_info)
            else:
                # No type inferred, use the TypeVar itself
                concrete_args.append(GenericInfo(origin=type_param))
        
        return concrete_args

    def _extract_typevar_bindings(self, annotation: Any, value: Any, typevar_to_types: Dict[TypeVar, Set[Any]]):
        """Extract TypeVar bindings from a field annotation and its corresponding value."""
        if isinstance(annotation, TypeVar):
            value_type = type(value)
            typevar_to_types[annotation].add(value_type)
        else:
            runtime_info = get_instance_generic_info(value)
            self._unify_annotation_with_runtime(annotation, runtime_info, typevar_to_types)

    def _unify_annotation_with_runtime(self, annotation: Any, runtime_info: Any, typevar_to_types: Dict[TypeVar, Set[Any]]):
        """Unify an annotation structure with runtime type information to extract TypeVar bindings."""
        ann_origin = get_origin(annotation) or annotation
        ann_args = get_args(annotation)
        
        if isinstance(annotation, TypeVar):
            if runtime_info.is_generic and runtime_info.resolved_type:
                typevar_to_types[annotation].add(runtime_info.resolved_type)
            else:
                typevar_to_types[annotation].add(runtime_info.origin)
        elif ann_origin == runtime_info.origin:
            # Same container type, unify arguments
            if ann_args and runtime_info.concrete_args:
                min_args = min(len(ann_args), len(runtime_info.concrete_args))
                for i in range(min_args):
                    ann_arg = ann_args[i]
                    runtime_arg = runtime_info.concrete_args[i]
                    
                    if isinstance(ann_arg, TypeVar):
                        self._bind_typevar_to_runtime(ann_arg, runtime_arg, typevar_to_types)
                    elif hasattr(ann_arg, "__origin__") or hasattr(ann_arg, "__args__"):
                        # Nested structure, recurse
                        self._unify_annotation_with_runtime(ann_arg, runtime_arg, typevar_to_types)
        elif ann_args and not runtime_info.is_generic:
            # Annotation is generic but runtime isn't
            for ann_arg in ann_args:
                if isinstance(ann_arg, TypeVar):
                    typevar_to_types[ann_arg].add(runtime_info.origin)

    def _bind_typevar_to_runtime(self, typevar: TypeVar, runtime_arg: GenericInfo, typevar_to_types: Dict[TypeVar, Set[Any]]):
        """Bind a TypeVar to a runtime type argument."""
        if runtime_arg.origin is Union and runtime_arg.concrete_args:
            # If runtime type is a Union, add all member types individually
            for union_member in runtime_arg.concrete_args:
                if union_member.is_generic and union_member.resolved_type:
                    typevar_to_types[typevar].add(union_member.resolved_type)
                else:
                    typevar_to_types[typevar].add(union_member.origin)
        elif runtime_arg.is_generic and runtime_arg.resolved_type:
            typevar_to_types[typevar].add(runtime_arg.resolved_type)
        else:
            typevar_to_types[typevar].add(runtime_arg.origin)


class BuiltinExtractor(GenericExtractor):
    """Extractor for built-in generic types like list, dict, tuple, set."""

    _BUILTIN_ORIGINS = frozenset({
        list, dict, tuple, set,
        List, Dict, Tuple, Set,
    })

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in self._BUILTIN_ORIGINS

    def can_handle_instance(self, instance: Any) -> bool:
        return isinstance(instance, (list, dict, tuple, set))

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        origin = get_origin(annotation)
        args = get_args(annotation)

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        origin = type(instance)
        concrete_args = self._infer_args_from_content(instance)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def _infer_args_from_content(self, instance: Any) -> List[GenericInfo]:
        """Infer type arguments from instance content."""
        if isinstance(instance, list) and instance:
            element_types = {get_instance_generic_info(item) for item in instance}
            return [GenericInfo.make_union_if_needed(element_types)]
        elif isinstance(instance, dict) and instance:
            key_types = {get_instance_generic_info(k) for k in instance.keys()}
            value_types = {get_instance_generic_info(v) for v in instance.values()}
            return [
                GenericInfo.make_union_if_needed(key_types), 
                GenericInfo.make_union_if_needed(value_types)
            ]
        elif isinstance(instance, tuple) and instance:
            return [get_instance_generic_info(item) for item in instance]
        elif isinstance(instance, set) and instance:
            element_types = {get_instance_generic_info(item) for item in instance}
            return [GenericInfo.make_union_if_needed(element_types)]
        return []


class PydanticExtractor(FieldBasedExtractor):
    """Extractor for Pydantic generic models."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        return hasattr(annotation, "__pydantic_generic_metadata__")

    def can_handle_instance(self, instance: Any) -> bool:
        return hasattr(instance, "__pydantic_generic_metadata__")

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        if not hasattr(annotation, "__pydantic_generic_metadata__"):
            return GenericInfo()

        metadata = annotation.__pydantic_generic_metadata__

        if metadata.get("origin"):
            # Specialized annotation (e.g., PydanticBox[int])
            origin = metadata["origin"]
            args = metadata.get("args", ())
            concrete_args = [get_generic_info(arg) for arg in args]
        else:
            origin = annotation
            original_type_params = self._get_original_type_parameters(annotation)
            concrete_args = [
                GenericInfo(origin=type_param) 
                for type_param in original_type_params
            ]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not hasattr(instance, "__pydantic_generic_metadata__"):
            return GenericInfo()

        instance_class = type(instance)
        metadata = instance_class.__pydantic_generic_metadata__

        if metadata.get("origin"):
            # Specialized class (e.g., PydanticBox[int])
            origin = metadata["origin"]
            args = metadata.get("args", ())
            concrete_args = [get_generic_info(arg) for arg in args]
        else:
            # Unparameterized base class (e.g., PydanticBox)
            origin = instance_class
            concrete_args = self._infer_from_field_values(instance)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def _infer_from_field_values(self, instance: Any) -> List[GenericInfo]:
        """Infer concrete type arguments from actual field values in a Pydantic instance."""
        try:
            original_type_params = self._get_original_type_parameters(type(instance))
            if not original_type_params:
                return []
            
            field_annotations = self._get_field_annotations(type(instance))
            if not field_annotations:
                return []
            
            field_values = {
                field_name: getattr(instance, field_name)
                for field_name in field_annotations.keys()
                if hasattr(instance, field_name)
            }
            
            return self._map_typevars_to_inferred_types(
                original_type_params, field_annotations, field_values
            )
            
        except (AttributeError, KeyError, TypeError):
            return []

    def _get_field_annotations(self, pydantic_class: Any) -> Dict[str, Any]:
        """Get field annotations from a Pydantic model."""
        return getattr(pydantic_class, "__annotations__", {})


class UnionExtractor(GenericExtractor):
    """Extractor for Union types (both typing.Union and types.UnionType)."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        union_type = getattr(types, "UnionType", None)
        return origin is Union or (union_type and origin is union_type)

    def can_handle_instance(self, instance: Any) -> bool:
        return False  # Instances don't have Union types directly

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract Union type information."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Union types don't have instances directly."""
        return GenericInfo(origin=type(instance))


class DataclassExtractor(FieldBasedExtractor):
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

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        if not is_dataclass(instance):
            return GenericInfo()

        origin = type(instance)

        # Check for __orig_class__ which preserves concrete type info
        if hasattr(instance, "__orig_class__"):
            args = get_args(instance.__orig_class__)
            concrete_args = [get_generic_info(arg) for arg in args]
        else:
            concrete_args = self._infer_from_field_values(instance)

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def _infer_from_field_values(self, instance: Any) -> List[GenericInfo]:
        """Infer concrete type arguments from actual field values in a dataclass instance."""
        try:
            original_type_params = self._get_original_type_parameters(type(instance))
            if not original_type_params:
                return []
            
            dataclass_fields = fields(instance)
            if not dataclass_fields:
                return []
            
            field_annotations = {field_info.name: field_info.type for field_info in dataclass_fields}
            field_values = {
                field_name: getattr(instance, field_name)
                for field_name in field_annotations.keys()
                if hasattr(instance, field_name)
            }
            
            return self._map_typevars_to_inferred_types(
                original_type_params, field_annotations, field_values
            )
            
        except (AttributeError, KeyError, TypeError):
            return []


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
        """Extract generic type information from an annotation."""
        if isinstance(annotation, TypeVar):
            return GenericInfo(origin=annotation)

        for extractor in self.extractors:
            if extractor.can_handle_annotation(annotation):
                return extractor.extract_from_annotation(annotation)

        # Fallback for non-generic types
        return GenericInfo(origin=annotation)

    def get_instance_generic_info(self, instance: Any) -> GenericInfo:
        """Extract generic type information from an instance."""
        for extractor in self.extractors:
            if extractor.can_handle_instance(instance):
                return extractor.extract_from_instance(instance)

        # Fallback for non-generic instances
        return GenericInfo(origin=type(instance))

    def get_type_parameters(self, annotation: Any) -> List[TypeVar]:
        """Extract TypeVar parameters from any generic annotation."""
        info = self.get_generic_info(annotation)
        return info.type_params

    def get_concrete_args(self, annotation: Any) -> List[GenericInfo]:
        """Extract concrete type arguments from any generic annotation."""
        info = self.get_generic_info(annotation)
        return info.concrete_args

    def get_instance_concrete_args(self, instance: Any) -> List[GenericInfo]:
        """Extract concrete type arguments from any generic instance."""
        info = self.get_instance_generic_info(instance)
        return info.concrete_args

    def get_generic_origin(self, annotation: Any) -> Any:
        """Get the generic origin from any annotation."""
        info = self.get_generic_info(annotation)
        return info.origin

    def is_generic_type(self, annotation: Any) -> bool:
        """Check if an annotation represents a generic type."""
        info = self.get_generic_info(annotation)
        return info.is_generic

    def get_resolved_type(self, annotation: Any) -> Any:
        """Get the fully resolved/materialized type from an annotation."""
        info = self.get_generic_info(annotation)
        return info.resolved_type

    def extract_all_typevars(self, annotation: Any) -> List[TypeVar]:
        """Recursively extract all TypeVars from a nested annotation structure."""
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
