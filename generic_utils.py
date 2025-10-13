"""
Utilities for extracting type information from generic annotations and instances.

This module provides structural extraction of generic type information, offering
a consistent interface for working with generic types across different systems
(built-in generics, Pydantic models, dataclasses).

Key concepts:
- concrete_args: The actual type arguments as GenericInfo objects  
- type_params: TypeVars extracted from the concrete arguments
- origin: The base generic type (e.g., list for list[int])
- resolved_type: The fully materialized type
"""

import functools
import typing
import types
from typing import Any, Dict, List, Set, TypeVar, Tuple, Union, get_args, get_origin, get_type_hints
from dataclasses import dataclass, field, is_dataclass, fields
from abc import ABC, abstractmethod


def is_union_type(origin: Any) -> bool:
    """Check if origin represents a Union type (handles both typing.Union and types.UnionType)."""
    union_type = getattr(types, 'UnionType', None)
    return origin is Union or (union_type and origin is union_type)


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
            try:
                return self.origin[*resolved_args]
            except (TypeError, AttributeError):
                # Some origins don't support subscription, return as-is
                return self.origin

    def _is_union_origin(self) -> bool:
        """Check if origin is a Union type."""
        return is_union_type(self.origin)
        
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

    @abstractmethod
    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs for type inference.
        
        Returns empty list if this extractor doesn't handle the annotation-instance pair.
        """
    
    def _build_typevar_substitution_map(self, annotation_info: GenericInfo) -> Dict[TypeVar, GenericInfo]:
        """Build a map from TypeVars to their concrete substitutions.
        
        Shared helper for extractors that need TypeVar substitution.
        Only creates mappings when TypeVars are bound to non-TypeVar types.
        """
        if not annotation_info.concrete_args:
            return {}
        
        # Get original TypeVars from the class definition
        original_typevars = self._get_original_type_parameters(annotation_info.origin)
        if not original_typevars:
            return {}
        
        # Map each TypeVar to its concrete arg, but skip identity mappings (A -> A)
        typevar_map = {}
        for typevar, concrete_arg in zip(original_typevars, annotation_info.concrete_args):
            # Only add mapping if concrete_arg is not the same TypeVar (avoid A -> A)
            if not (isinstance(concrete_arg.origin, TypeVar) and concrete_arg.origin == typevar):
                typevar_map[typevar] = concrete_arg
        
        return typevar_map
    
    def _substitute_typevars_in_generic_info(
        self, 
        generic_info: GenericInfo, 
        typevar_map: Dict[TypeVar, GenericInfo]
    ) -> GenericInfo:
        """Substitute TypeVars in a GenericInfo structure.
        
        Shared helper for extractors that need TypeVar substitution.
        """
        # If this is a TypeVar, substitute it
        if isinstance(generic_info.origin, TypeVar) and generic_info.origin in typevar_map:
            return typevar_map[generic_info.origin]
        
        # If no concrete args, return as-is
        if not generic_info.concrete_args:
            return generic_info
        
        # Recursively substitute in concrete args
        substituted_args = [
            self._substitute_typevars_in_generic_info(arg, typevar_map)
            for arg in generic_info.concrete_args
        ]
        
        # Return new GenericInfo with substituted args
        return GenericInfo(origin=generic_info.origin, concrete_args=substituted_args)
    
    @abstractmethod
    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a class definition.
        
        Implemented by subclasses based on their specific metadata mechanisms.
        """


class BuiltinExtractor(GenericExtractor):
    """Extractor for built-in generic types like list, dict, tuple, set."""

    _BUILTIN_ORIGINS = frozenset({
        list, dict, tuple, set,
        List, Dict, Tuple, Set,
    })
    
    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Built-in types don't have TypeVar parameters in their definitions."""
        return []

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in self._BUILTIN_ORIGINS

    def can_handle_instance(self, instance: Any) -> bool:
        return isinstance(instance, (list, dict, tuple, set))

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a built-in type annotation."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from a built-in type instance."""
        return GenericInfo(origin=type(instance))

    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs from built-in containers."""
        if instance is None:
            return []
        
        annotation_info = self.extract_from_annotation(annotation)
        if not annotation_info.concrete_args:
            return []
        
        pairs = []
        
        # Handle list
        if annotation_info.origin in (list, List):
            if len(annotation_info.concrete_args) == 1 and isinstance(instance, list):
                element_generic_info = annotation_info.concrete_args[0]
                for value in instance:
                    pairs.append((element_generic_info, value))
        
        # Handle set
        elif annotation_info.origin in (set, Set):
            if len(annotation_info.concrete_args) == 1 and isinstance(instance, set):
                element_generic_info = annotation_info.concrete_args[0]
                for value in instance:
                    pairs.append((element_generic_info, value))
        
        # Handle dict
        elif annotation_info.origin in (dict, Dict):
            if len(annotation_info.concrete_args) == 2 and isinstance(instance, dict):
                key_generic_info, value_generic_info = annotation_info.concrete_args
                # Add key mappings
                for key in instance.keys():
                    pairs.append((key_generic_info, key))
                # Add value mappings
                for value in instance.values():
                    pairs.append((value_generic_info, value))
        
        # Handle tuple
        elif annotation_info.origin in (tuple, Tuple):
            if isinstance(instance, tuple):
                # Handle variable length tuple: tuple[A, ...]
                if len(annotation_info.concrete_args) == 2 and annotation_info.concrete_args[1].origin is ...:
                    element_generic_info = annotation_info.concrete_args[0]
                    for value in instance:
                        pairs.append((element_generic_info, value))
                # Handle fixed length tuple: tuple[A, B, C]
                else:
                    for i, value in enumerate(instance):
                        if i < len(annotation_info.concrete_args):
                            pairs.append((annotation_info.concrete_args[i], value))
        
        return pairs


class PydanticExtractor(GenericExtractor):
    """Extractor for Pydantic generic models."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        return hasattr(annotation, "__pydantic_generic_metadata__")

    def can_handle_instance(self, instance: Any) -> bool:
        return hasattr(instance, "__pydantic_generic_metadata__")

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a Pydantic type annotation."""
        if not hasattr(annotation, "__pydantic_generic_metadata__"):
            return GenericInfo()

        metadata = annotation.__pydantic_generic_metadata__

        if metadata.get("origin"):
            # Specialized annotation (e.g., PydanticBox[int])
            origin = metadata["origin"]
            args = metadata.get("args", ())
            concrete_args = [get_generic_info(arg) for arg in args]
        else:
            # Unparameterized base class - extract TypeVars from class definition
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
        """Extract generic information from a Pydantic model instance."""
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
            # Unparameterized base class
            origin = instance_class
            concrete_args = []

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a class definition."""
        for base in getattr(dataclass_class, "__orig_bases__", []):
            if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                origin = get_origin(base)
                if origin and hasattr(origin, "__name__") and "Generic" in str(origin):
                    args = get_args(base)
                    return [arg for arg in args if isinstance(arg, TypeVar)]
        return []

    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs from Pydantic model fields.
        
        Pydantic specializes field annotations in parameterized classes, so we can
        use annotation's fields directly without manual substitution:
        - Level3[A] → fields have TypeVar A
        - Level3[bool] → fields have bool
        - Level3[List[B]] → fields have List[B]
        """
        if instance is None or not hasattr(instance, "__pydantic_fields__"):
            return []
        
        # Use annotation's fields directly - Pydantic already specializes them
        if not hasattr(annotation, "__pydantic_fields__"):
            return []
        
        pairs = []
        for field_name, field_info in annotation.__pydantic_fields__.items():
            field_value = getattr(instance, field_name)
            # Map each field to its annotation (already specialized by Pydantic)
            field_generic_info = get_generic_info(field_info.annotation)
            pairs.append((field_generic_info, field_value))
        
        return pairs


class UnionExtractor(GenericExtractor):
    """Extractor for Union types (both typing.Union and types.UnionType)."""
    
    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Union types don't have TypeVar parameters in their definitions."""
        return []

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

    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Union types don't have direct instances.
        
        Matching an instance to a Union alternative requires type-checking logic
        that belongs in the unification engine, not in structural extraction.
        The unification engine handles this via _handle_union_constraints.
        """
        return []


class DataclassExtractor(GenericExtractor):
    """Extractor for dataclass generic types."""

    def can_handle_annotation(self, annotation: Any) -> bool:
        origin = get_origin(annotation) or annotation
        return is_dataclass(origin) and hasattr(origin, "__orig_bases__")

    def can_handle_instance(self, instance: Any) -> bool:
        return is_dataclass(instance)

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a dataclass type annotation."""
        origin = get_origin(annotation) or annotation
        args = get_args(annotation)

        if not is_dataclass(origin):
            return GenericInfo()

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from a dataclass instance."""
        if not is_dataclass(instance):
            return GenericInfo()

        origin = type(instance)

        # Check for __orig_class__ which preserves concrete type info
        if hasattr(instance, "__orig_class__"):
            args = get_args(instance.__orig_class__)
            concrete_args = [get_generic_info(arg) for arg in args]
        else:
            # Use the class type without type arguments
            concrete_args = []

        return GenericInfo(
            origin=origin, concrete_args=concrete_args
        )

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a class definition."""
        for base in getattr(dataclass_class, "__orig_bases__", []):
            if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                origin = get_origin(base)
                if origin and hasattr(origin, "__name__") and "Generic" in str(origin):
                    args = get_args(base)
                    return [arg for arg in args if isinstance(arg, TypeVar)]
        return []
    
    def _build_inheritance_aware_substitution(self, annotation_info: GenericInfo, instance: Any) -> Dict[TypeVar, GenericInfo]:
        """Build substitution map for inherited fields with potentially swapped TypeVars.
        
        Handles cases like:
            class HasA(Generic[A, B]): ...
            class HasB(HasA[B, A], Generic[A, B]): ...  # Swapped!
            
        When extracting fields from HasB that are inherited from HasA, we need to map
        HasA's TypeVars through the inheritance chain to HasB's TypeVars.
        
        Args:
            annotation_info: The annotation (e.g., HasB[C, D])
            instance: The instance (e.g., HasB[int, str] instance)
            
        Returns:
            Dict mapping field TypeVars (from parent classes) to annotation TypeVars
        """
        # Start with simple substitution for the annotation
        typevar_map = self._build_typevar_substitution_map(annotation_info)
        
        instance_class = instance.__orig_class__
        annotation_class = annotation_info.origin
        instance_origin = get_origin(instance_class) or instance_class
        
        # Walk through all parent dataclasses to build substitution maps
        if not hasattr(instance_origin, '__orig_bases__'):
            return typevar_map
        
        # For each parent class, map its TypeVars to the annotation's TypeVars
        for base in instance_origin.__orig_bases__:
            base_origin = get_origin(base) or base
            if base_origin == annotation_class or not is_dataclass(base_origin):
                continue  # Skip self and non-dataclasses
            
            # Found a parent dataclass! e.g., base = HasA[B, A] where B, A are from HasB
            base_args = get_args(base)  # [B, A] (TypeVars from instance class)
            instance_params = getattr(instance_origin, '__parameters__', ())  # (A, B) from HasB
            
            # Get the parent class's original TypeVars
            parent_class_params = self._get_original_type_parameters(base_origin)  # [A, B] from HasA
            
            # Map each parent TypeVar to the annotation's TypeVar
            # parent_class_params[i] (HasA's TypeVar) appears as base_args[i] (HasB's TypeVar reference)
            # We need to find where base_args[i] appears in instance_params and use annotation_info.concrete_args
            
            for i, parent_tv in enumerate(parent_class_params):
                if i < len(base_args):
                    base_arg = base_args[i]  # This is a TypeVar from instance class (e.g., B from HasB)
                    
                    if isinstance(base_arg, TypeVar):
                        # Find where this TypeVar appears in the instance's parameters
                        try:
                            param_idx = list(instance_params).index(base_arg)
                            # Map parent TypeVar to annotation's corresponding TypeVar
                            if param_idx < len(annotation_info.concrete_args):
                                typevar_map[parent_tv] = annotation_info.concrete_args[param_idx]
                        except ValueError:
                            pass
                    else:
                        # base_arg is a concrete type
                        typevar_map[parent_tv] = get_generic_info(base_arg)
        
        return typevar_map

    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs from dataclass fields.
        
        IMPORTANT: Only extracts fields defined in the annotation class, not inherited fields.
        This ensures that when matching HasA[A] against a HasBoth instance, we only
        get HasA's fields, avoiding TypeVar shadowing issues.
        
        Also handles inheritance with swapped TypeVars like HasB(HasA[B, A], Generic[A, B]).
        """
        if instance is None or not is_dataclass(instance):
            return []
        
        annotation_info = self.extract_from_annotation(annotation)
        if not is_dataclass(annotation_info.origin):
            return []
        
        # Check if we need inheritance substitution (annotation class != instance class)
        annotation_class = annotation_info.origin
        instance_class_origin = get_origin(getattr(instance, '__orig_class__', type(instance))) or type(instance)
        
        # Build TypeVar substitution map
        # Check if this class has parent classes with generic parameters
        has_generic_parents = False
        if hasattr(annotation_class, '__orig_bases__'):
            for base in annotation_class.__orig_bases__:
                base_origin = get_origin(base) or base
                if base_origin != annotation_class and is_dataclass(base_origin):
                    # Has a dataclass parent
                    has_generic_parents = True
                    break
        
        if has_generic_parents and hasattr(instance, '__orig_class__'):
            # This class inherits from generic dataclasses - need to track inheritance substitution
            # Example: HasB[C, D] where HasB inherits HasA[B, A] (swapped)
            typevar_map = self._build_inheritance_aware_substitution(annotation_info, instance)
        else:
            # No generic parents or no __orig_class__ - simple substitution
            typevar_map = self._build_typevar_substitution_map(annotation_info)
        
        # Get resolved field types (resolves ForwardRefs)
        # Build localns with the class itself for ForwardRef resolution in local scopes
        try:
            import sys
            module = sys.modules.get(annotation_info.origin.__module__)
            globalns = vars(module) if module else {}
            # Include the class itself in localns to resolve self-referential ForwardRefs
            localns = {annotation_info.origin.__name__: annotation_info.origin}
            field_hints = get_type_hints(annotation_info.origin, globalns=globalns, localns=localns)
        except Exception:
            # Fallback to raw field types if get_type_hints fails
            field_hints = {}
        
        pairs = []
        # CRITICAL FIX: Iterate over annotation class's fields only, not instance's fields
        # This prevents extracting inherited fields that might use shadowed TypeVar names
        for dataclass_field in fields(annotation_info.origin):
            # Check if the instance actually has this field (it should, due to inheritance)
            if not hasattr(instance, dataclass_field.name):
                continue
                
            field_value = getattr(instance, dataclass_field.name)
            
            # Use resolved field type if available, otherwise use raw type
            field_type = field_hints.get(dataclass_field.name, dataclass_field.type)
            field_generic_info = get_generic_info(field_type)
            
            # Substitute TypeVars to handle re-parameterization
            # Example: Field is 'A' but annotation uses 'B', map A → B
            if typevar_map:
                field_generic_info = self._substitute_typevars_in_generic_info(field_generic_info, typevar_map)
            
            pairs.append((field_generic_info, field_value))
        
        return pairs


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

    def get_annotation_value_pairs(self, annotation: Any, instance: Any) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs for type inference.
        
        This provides a unified interface for all container types:
        - For list[A] with [1, 2, 3] → [(GenericInfo(origin=A), 1), (GenericInfo(origin=A), 2), (GenericInfo(origin=A), 3)]
        - For dict[A, B] with {"key": 42} → [(GenericInfo(origin=A), "key"), (GenericInfo(origin=B), 42)]  
        - For DataClass[A] with instance → [(GenericInfo(origin=A), field_val1), (GenericInfo(origin=A), field_val2)]
        
        Args:
            annotation: The type annotation (e.g., list[A], dict[A, B], DataClass[A])
            instance: The concrete instance to extract values from
            
        Returns:
            List of (GenericInfo, value) pairs for type inference
        """
        if instance is None:
            return []
        
        # Get annotation structure
        annotation_info = self.get_generic_info(annotation)
        if not annotation_info.concrete_args:
            return []  # No type parameters to bind
        
        # Try each extractor
        for extractor in self.extractors:
            if extractor.can_handle_annotation(annotation):
                pairs = extractor.get_annotation_value_pairs(annotation, instance)
                if pairs:
                    return pairs
        
        # Fallback for custom generic objects with __dict__
        if hasattr(instance, '__dict__') and annotation_info.concrete_args:
            pairs = []
            first_typevar_info = annotation_info.concrete_args[0]
            for key, value in instance.__dict__.items():
                # Skip special attributes that shouldn't be used for type inference
                if not key.startswith('__'):
                    pairs.append((first_typevar_info, value))
            return pairs
        
        return []


def create_union_if_needed(types_set: set) -> Any:
    """Create a Union type if needed, or return single type.
    
    Uses modern union syntax (int | str) for Python 3.10+ compatibility.
    """
    if len(types_set) == 1:
        return list(types_set)[0]
    elif len(types_set) > 1:
        try:
            # Use modern union syntax for better readability and performance
            result = types_set.pop()
            for elem_type in types_set:
                result = result | elem_type
            return result
        except TypeError:
            # Fallback to typing.Union for edge cases where | operator doesn't work
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


def get_annotation_value_pairs(annotation: Any, instance: Any) -> List[Tuple["GenericInfo", Any]]:
    """Extract (GenericInfo, value) pairs for type inference."""
    return generic_utils.get_annotation_value_pairs(annotation, instance)
