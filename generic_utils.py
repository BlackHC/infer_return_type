"""
Generic Type Utilities for Structural Type Information Extraction.

This module provides a unified interface for extracting structural type information
from generic annotations and instances across different type systems. It supports
built-in generics (list, dict, tuple, set), Pydantic models, dataclasses, and Union types.

The core abstraction is GenericInfo, which represents generic types in a consistent
structural format that can be used for type inference, constraint solving, and
unification algorithms.

Key Concepts:
    GenericInfo: Structural representation of generic types with:
        - origin: The base generic type (e.g., list for list[int])
        - concrete_args: Type arguments as GenericInfo objects
        - type_params: TypeVars extracted from concrete arguments
        - resolved_type: The fully materialized type
    
    Extractors: Type-system-specific handlers that implement the GenericExtractor
    interface to extract type information from different generic systems.
    
    Annotation-Value Pairs: (GenericInfo, value) tuples used for type inference
    by mapping type structure to runtime values.

Architecture:
    The module uses a plugin-based architecture with specialized extractors:
    - BuiltinExtractor: Handles list, dict, tuple, set types
    - PydanticExtractor: Handles Pydantic generic models
    - DataclassExtractor: Handles dataclass generic types
    - UnionExtractor: Handles Union types (typing.Union and types.UnionType)
    
    GenericTypeUtils provides the unified interface that delegates to appropriate
    extractors based on the type being processed.

Example:
    >>> from typing import List, TypeVar
    >>> from generic_utils import get_generic_info, get_annotation_value_pairs
    >>> 
    >>> A = TypeVar('A')
    >>> info = get_generic_info(List[A])
    >>> print(info.origin)  # <class 'list'>
    >>> print(len(info.concrete_args))  # 1
    >>> 
    >>> pairs = get_annotation_value_pairs(List[A], [1, 2, 3])
    >>> print(len(pairs))  # 3 (one pair per list element)
"""

import functools
import typing
import types
from typing import (
    Any,
    Dict,
    List,
    Set,
    TypeVar,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from dataclasses import dataclass, field, is_dataclass, fields
from abc import ABC, abstractmethod


def is_union_type(origin: Any) -> bool:
    """Check if origin represents a Union type.
    
    Handles both typing.Union and types.UnionType (Python 3.10+).
    
    Args:
        origin: The type origin to check
        
    Returns:
        True if the origin represents a Union type, False otherwise
        
    Example:
        >>> is_union_type(Union)
        True
        >>> is_union_type(int)
        False
        >>> is_union_type(int | str)  # Python 3.10+
        True
    """
    return origin is Union or origin is types.UnionType


@dataclass(frozen=True, kw_only=True)
class GenericInfo:
    """Structural representation of generic type information.
    
    GenericInfo provides a unified structural representation of generic types
    that abstracts away differences between type systems (built-ins, Pydantic,
    dataclasses). It contains both the raw structural information and computed
    derived properties for efficient type inference.
    
    Attributes:
        origin: The base generic type (e.g., list for list[int])
        concrete_args: Type arguments as GenericInfo objects
        type_params: TypeVars extracted from concrete_args (computed)
        
    Properties:
        is_generic: Whether this type has generic information
        resolved_type: The fully materialized type (cached)
        resolved_concrete_args: The materialized concrete arguments
        
    Example:
        >>> from typing import List, TypeVar
        >>> A = TypeVar('A')
        >>> info = GenericInfo(origin=list, concrete_args=[GenericInfo(origin=A)])
        >>> print(info.is_generic)  # True
        >>> print(info.type_params)  # [A]
        >>> print(info.resolved_type)  # list[A] (if A is bound)
    """

    origin: Any = None
    concrete_args: List["GenericInfo"] = field(default_factory=list)
    type_params: List[TypeVar] = field(init=False)

    def __post_init__(self):
        """Compute derived fields after initialization.
        
        Automatically computes type_params from concrete_args after the
        dataclass is initialized. This ensures type_params is always
        up-to-date with the concrete_args.
        """
        object.__setattr__(self, "type_params", self._compute_type_params())

    @property
    def is_generic(self) -> bool:
        """Whether this type has generic information.
        
        Returns True if this GenericInfo represents a generic type with
        type parameters, False otherwise.
        
        Returns:
            True if concrete_args is non-empty, False otherwise
            
        Example:
            >>> GenericInfo(origin=list, concrete_args=[GenericInfo(origin=int)]).is_generic
            True
            >>> GenericInfo(origin=int).is_generic
            False
        """
        return bool(self.concrete_args)

    def _compute_type_params(self) -> List[TypeVar]:
        """Compute TypeVars from concrete_args and their nested type_params.
        
        Recursively extracts all TypeVars from the concrete_args structure.
        This includes both direct TypeVars and TypeVars nested within
        other GenericInfo objects.
        
        Returns:
            List of unique TypeVars found in the structure
            
        Example:
            >>> A, B = TypeVar('A'), TypeVar('B')
            >>> info = GenericInfo(
            ...     origin=dict,
            ...     concrete_args=[
            ...         GenericInfo(origin=A),
            ...         GenericInfo(origin=list, concrete_args=[GenericInfo(origin=B)])
            ...     ]
            ... )
            >>> info.type_params  # [A, B]
        """
        seen = set()
        for arg in self.concrete_args:
            if isinstance(arg.origin, TypeVar):
                seen.add(arg.origin)
            else:
                seen.update(arg.type_params)
        return list(seen)

    @functools.cached_property
    def resolved_type(self) -> Any:
        """The fully materialized type using origin[*resolved_args].
        
        Recursively resolves all GenericInfo objects in concrete_args to
        their actual types and constructs the final type using the origin.
        
        Special handling:
        - Union types: Creates unions using modern syntax (int | str)
        - Tuple types: Handles both fixed and variable-length tuples
        - Fallback: Returns origin if subscription fails
        
        Returns:
            The fully materialized type (e.g., list[int], dict[str, int])
            
        Example:
            >>> info = GenericInfo(
            ...     origin=list,
            ...     concrete_args=[GenericInfo(origin=int)]
            ... )
            >>> info.resolved_type  # list[int]
        """
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
        """Check if origin is a Union type.
        
        Returns:
            True if the origin represents a Union type
        """
        return is_union_type(self.origin)

    @functools.cached_property
    def resolved_concrete_args(self) -> List[Any]:
        """The fully materialized concrete arguments.
        
        Returns a list of resolved types from concrete_args, providing
        direct access to the materialized type arguments without the
        GenericInfo wrapper.
        
        Returns:
            List of resolved types, or empty list if no concrete_args
            
        Example:
            >>> info = GenericInfo(
            ...     origin=dict,
            ...     concrete_args=[
            ...         GenericInfo(origin=str),
            ...         GenericInfo(origin=int)
            ...     ]
            ... )
            >>> info.resolved_concrete_args  # [str, int]
        """
        return (
            [arg.resolved_type for arg in self.concrete_args]
            if self.concrete_args
            else []
        )

    def __eq__(self, other):
        """Check equality based on origin and resolved_type.
        
        Two GenericInfo objects are equal if they have the same origin
        and resolve to the same type. This allows GenericInfo objects
        to be used in sets and as dictionary keys.
        
        Args:
            other: The object to compare with
            
        Returns:
            True if objects are equal, False otherwise
        """
        if not isinstance(other, GenericInfo):
            return False
        return self.origin == other.origin and self.resolved_type == other.resolved_type

    def __hash__(self):
        """Make GenericInfo hashable based on origin and resolved_type.
        
        Enables GenericInfo objects to be used in sets and as dictionary
        keys. Falls back to string representation if resolved_type is
        not hashable.
        
        Returns:
            Hash value for the object
        """
        try:
            return hash((self.origin, self.resolved_type))
        except TypeError:
            return hash((self.origin, str(self.resolved_type)))


class GenericExtractor(ABC):
    """Abstract base class for type-system-specific generic extractors.
    
    GenericExtractor defines the interface that all type-system-specific
    extractors must implement. Each extractor handles a specific type system
    (built-ins, Pydantic, dataclasses, etc.) and provides methods to:
    - Check if it can handle a given annotation or instance
    - Extract GenericInfo from annotations and instances
    - Generate annotation-value pairs for type inference
    
    The plugin-based architecture allows the system to be extended
    to support new type systems by implementing this interface.
    """

    @abstractmethod
    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation.
        
        Args:
            annotation: The type annotation to check
            
        Returns:
            True if this extractor can process the annotation
        """

    @abstractmethod
    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance.
        
        Args:
            instance: The runtime instance to check
            
        Returns:
            True if this extractor can process the instance
        """

    @abstractmethod
    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a type annotation.
        
        Args:
            annotation: The type annotation to extract from
            
        Returns:
            GenericInfo representing the structural type information
        """

    @abstractmethod
    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from an instance.
        
        Args:
            instance: The runtime instance to extract from
            
        Returns:
            GenericInfo representing the structural type information
        """

    @abstractmethod
    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs for type inference.
        
        This method is crucial for type inference as it maps type structure
        to runtime values. For example, for List[A] with [1, 2, 3], it would
        return [(GenericInfo(origin=A), 1), (GenericInfo(origin=A), 2), (GenericInfo(origin=A), 3)].
        
        Args:
            annotation: The type annotation
            instance: The runtime instance
            
        Returns:
            List of (GenericInfo, value) pairs, or empty list if this extractor
            doesn't handle the annotation-instance pair
        """

    def _build_typevar_substitution_map(
        self, annotation_info: GenericInfo
    ) -> Dict[TypeVar, GenericInfo]:
        """Build a map from TypeVars to their concrete substitutions.
        
        This is a shared helper for extractors that need TypeVar substitution.
        It maps each TypeVar in the annotation to its corresponding concrete
        type argument, but only when the concrete type is not the same TypeVar
        (avoiding identity mappings like A -> A).
        
        Args:
            annotation_info: The GenericInfo representing the annotation
            
        Returns:
            Dictionary mapping TypeVars to their concrete GenericInfo substitutions
            
        Example:
            For Box[A, B] with Box[int, str], returns {A: GenericInfo(origin=int), B: GenericInfo(origin=str)}
        """
        if not annotation_info.concrete_args:
            return {}

        # Get original TypeVars from the class definition
        original_typevars = self._get_original_type_parameters(annotation_info.origin)
        if not original_typevars:
            return {}

        # Map each TypeVar to its concrete arg, but skip identity mappings (A -> A)
        typevar_map = {}
        for typevar, concrete_arg in zip(
            original_typevars, annotation_info.concrete_args
        ):
            # Only add mapping if concrete_arg is not the same TypeVar (avoid A -> A)
            if not (
                isinstance(concrete_arg.origin, TypeVar)
                and concrete_arg.origin == typevar
            ):
                typevar_map[typevar] = concrete_arg

        return typevar_map

    def _substitute_typevars_in_generic_info(
        self, generic_info: GenericInfo, typevar_map: Dict[TypeVar, GenericInfo]
    ) -> GenericInfo:
        """Substitute TypeVars in a GenericInfo structure.
        
        This is a shared helper for extractors that need TypeVar substitution.
        It recursively traverses the GenericInfo structure and replaces TypeVars
        with their concrete substitutions from the typevar_map.
        
        Args:
            generic_info: The GenericInfo to substitute TypeVars in
            typevar_map: Dictionary mapping TypeVars to their substitutions
            
        Returns:
            New GenericInfo with TypeVars substituted
            
        Example:
            GenericInfo(origin=list, concrete_args=[GenericInfo(origin=A)]) with
            {A: GenericInfo(origin=int)} becomes GenericInfo(origin=list, concrete_args=[GenericInfo(origin=int)])
        """
        # If this is a TypeVar, substitute it
        if (
            isinstance(generic_info.origin, TypeVar)
            and generic_info.origin in typevar_map
        ):
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
        
        Different type systems store TypeVar parameters in different ways:
        - Dataclasses: In __orig_bases__ from Generic[...] inheritance
        - Pydantic: In __orig_bases__ from Generic[...] inheritance
        - Built-ins: Don't have TypeVar parameters
        
        Args:
            dataclass_class: The class to extract TypeVars from
            
        Returns:
            List of TypeVars defined in the class
        """


class BuiltinExtractor(GenericExtractor):
    """Extractor for built-in generic types.
    
    Handles Python's built-in generic types: list, dict, tuple, set and their
    typing module equivalents (List, Dict, Tuple, Set). These types use
    standard Python generics with __origin__ and __args__ attributes.
    
    Supported types:
        - list, List: Single type parameter for element type
        - dict, Dict: Two type parameters for key and value types
        - tuple, Tuple: Variable number of type parameters
        - set, Set: Single type parameter for element type
    """

    _BUILTIN_ORIGINS = frozenset(
        {
            list,
            dict,
            tuple,
            set,
            List,
            Dict,
            Tuple,
            Set,
        }
    )

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Built-in types don't have TypeVar parameters in their definitions.
        
        Built-in types like list, dict, tuple, set are not generic classes
        with TypeVar parameters. They are concrete types that can be
        parameterized with type arguments, but don't define their own TypeVars.
        
        Args:
            dataclass_class: Ignored for built-in types
            
        Returns:
            Empty list (built-ins have no TypeVar parameters)
        """
        return []

    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation.
        
        Args:
            annotation: The type annotation to check
            
        Returns:
            True if the annotation's origin is a built-in generic type
        """
        origin = get_origin(annotation)
        return origin in self._BUILTIN_ORIGINS

    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance.
        
        Args:
            instance: The runtime instance to check
            
        Returns:
            True if the instance is a built-in container type
        """
        return isinstance(instance, (list, dict, tuple, set))

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a built-in type annotation.
        
        Args:
            annotation: The built-in type annotation (e.g., list[int])
            
        Returns:
            GenericInfo with origin and concrete_args extracted
            
        Example:
            >>> extractor = BuiltinExtractor()
            >>> info = extractor.extract_from_annotation(list[int])
            >>> info.origin  # <class 'list'>
            >>> info.concrete_args[0].origin  # <class 'int'>
        """
        origin = get_origin(annotation)
        args = get_args(annotation)

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from a built-in type instance.
        
        For built-in types, we can only extract the runtime type, not
        the generic parameters, since Python doesn't preserve generic
        type information at runtime for built-in types.
        
        Args:
            instance: The runtime instance
            
        Returns:
            GenericInfo with origin set to the instance's type
            
        Example:
            >>> extractor = BuiltinExtractor()
            >>> info = extractor.extract_from_instance([1, 2, 3])
            >>> info.origin  # <class 'list'>
            >>> info.concrete_args  # [] (no generic info preserved)
        """
        return GenericInfo(origin=type(instance))

    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs from built-in containers.
        
        Maps each element in the container to its corresponding type parameter.
        For example, list[A] with [1, 2, 3] produces three pairs mapping
        each integer to the GenericInfo representing type parameter A.
        
        Args:
            annotation: The container type annotation
            instance: The runtime container instance
            
        Returns:
            List of (GenericInfo, value) pairs for type inference
            
        Example:
            >>> extractor = BuiltinExtractor()
            >>> pairs = extractor.get_annotation_value_pairs(list[A], [1, 2, 3])
            >>> len(pairs)  # 3
            >>> pairs[0][1]  # 1 (the value)
            >>> pairs[0][0].origin  # A (the TypeVar)
        """
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
                if (
                    len(annotation_info.concrete_args) == 2
                    and annotation_info.concrete_args[1].origin is ...
                ):
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
    """Extractor for Pydantic generic models.
    
    Handles Pydantic's generic model system which uses __pydantic_generic_metadata__
    to store generic type information. Pydantic models can be parameterized with
    type arguments and preserve this information through the metadata system.
    
    Features:
        - Extracts generic info from both parameterized and unparameterized models
        - Handles TypeVar substitution for field annotations
        - Supports inheritance with generic base classes
        - Uses Pydantic's specialized field annotations
    """

    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation.
        
        Args:
            annotation: The type annotation to check
            
        Returns:
            True if the annotation has Pydantic generic metadata
        """
        return hasattr(annotation, "__pydantic_generic_metadata__")

    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance.
        
        Args:
            instance: The runtime instance to check
            
        Returns:
            True if the instance has Pydantic generic metadata
        """
        return hasattr(instance, "__pydantic_generic_metadata__")

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a Pydantic type annotation.
        
        Handles both parameterized models (e.g., Box[int]) and unparameterized
        base classes (e.g., Box). For unparameterized classes, extracts
        TypeVars from the class definition.
        
        Args:
            annotation: The Pydantic model annotation
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> class Box(BaseModel, Generic[A]):
            ...     item: A
            >>> extractor = PydanticExtractor()
            >>> info = extractor.extract_from_annotation(Box[int])
            >>> info.origin  # Box
            >>> info.concrete_args[0].origin  # int
        """
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
                GenericInfo(origin=type_param) for type_param in original_type_params
            ]

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from a Pydantic model instance.
        
        Args:
            instance: The Pydantic model instance
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> box = Box[int](item=42)
            >>> extractor = PydanticExtractor()
            >>> info = extractor.extract_from_instance(box)
            >>> info.origin  # Box
            >>> info.concrete_args[0].origin  # int
        """
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

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a Pydantic class definition.
        
        Extracts TypeVars from __orig_bases__ by looking for Generic[...] inheritance.
        
        Args:
            dataclass_class: The Pydantic model class
            
        Returns:
            List of TypeVars defined in the class
        """
        for base in getattr(dataclass_class, "__orig_bases__", []):
            if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                origin = get_origin(base)
                if origin and hasattr(origin, "__name__") and "Generic" in str(origin):
                    args = get_args(base)
                    return [arg for arg in args if isinstance(arg, TypeVar)]
        return []

    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
        """Extract (GenericInfo, value) pairs from Pydantic model fields.
        
        Pydantic specializes field annotations in parameterized classes, so we can
        use annotation's fields directly without manual substitution:
        - Box[A] → fields have TypeVar A
        - Box[bool] → fields have bool
        - Box[List[B]] → fields have List[B]
        
        Args:
            annotation: The Pydantic model annotation
            instance: The Pydantic model instance
            
        Returns:
            List of (GenericInfo, value) pairs for type inference
            
        Example:
            >>> class Box(BaseModel, Generic[A]):
            ...     item: A
            >>> box = Box[int](item=42)
            >>> pairs = extractor.get_annotation_value_pairs(Box[A], box)
            >>> len(pairs)  # 1
            >>> pairs[0][0].origin  # A (the TypeVar)
            >>> pairs[0][1]  # 42 (the value)
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
    """Extractor for Union types (both typing.Union and types.UnionType).
    
    Handles Union types which represent alternative types. Union types don't
    have instances directly, but are used in annotations to specify that
    a value can be one of several types.
    
    Features:
        - Supports both typing.Union and types.UnionType (Python 3.10+)
        - Extracts union alternatives as concrete_args
        - No direct instance handling (unions are type-level constructs)
    """

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Union types don't have TypeVar parameters in their definitions.
        
        Union types are not generic classes with TypeVar parameters.
        They are type-level constructs that combine multiple types.
        
        Args:
            dataclass_class: Ignored for Union types
            
        Returns:
            Empty list (Union types have no TypeVar parameters)
        """
        return []

    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation.
        
        Args:
            annotation: The type annotation to check
            
        Returns:
            True if the annotation is a Union type
        """
        origin = get_origin(annotation)
        return origin is Union or origin is types.UnionType

    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance.
        
        Args:
            instance: The runtime instance to check
            
        Returns:
            False (instances don't have Union types directly)
        """
        return False  # Instances don't have Union types directly

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract Union type information.
        
        Args:
            annotation: The Union type annotation
            
        Returns:
            GenericInfo with Union origin and alternatives as concrete_args
            
        Example:
            >>> extractor = UnionExtractor()
            >>> info = extractor.extract_from_annotation(int | str)
            >>> info.origin  # Union or types.UnionType
            >>> len(info.concrete_args)  # 2
            >>> info.concrete_args[0].origin  # int
            >>> info.concrete_args[1].origin  # str
        """
        origin = get_origin(annotation)
        args = get_args(annotation)

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Union types don't have instances directly.
        
        Args:
            instance: The runtime instance
            
        Returns:
            GenericInfo with the instance's type as origin
        """
        return GenericInfo(origin=type(instance))

    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
        """Union types don't have direct instances.

        Matching an instance to a Union alternative requires type-checking logic
        that belongs in the unification engine, not in structural extraction.
        The unification engine handles this via _handle_union_constraints.
        
        Args:
            annotation: The Union type annotation
            instance: The runtime instance
            
        Returns:
            Empty list (Union matching handled by unification engine)
        """
        return []


class DataclassExtractor(GenericExtractor):
    """Extractor for dataclass generic types.
    
    Handles dataclass-based generic types that inherit from typing.Generic.
    These classes use __orig_bases__ to store generic type information and
    __orig_class__ on instances to preserve concrete type arguments.
    
    Features:
        - Extracts generic info from dataclass annotations and instances
        - Handles TypeVar substitution for field annotations
        - Supports inheritance with swapped TypeVars (e.g., HasB(HasA[B, A]))
        - Uses dataclass fields() for field extraction
    """

    def can_handle_annotation(self, annotation: Any) -> bool:
        """Check if this extractor can handle the given annotation.
        
        Args:
            annotation: The type annotation to check
            
        Returns:
            True if the annotation is a dataclass with generic bases
        """
        origin = get_origin(annotation) or annotation
        return is_dataclass(origin) and hasattr(origin, "__orig_bases__")

    def can_handle_instance(self, instance: Any) -> bool:
        """Check if this extractor can handle the given instance.
        
        Args:
            instance: The runtime instance to check
            
        Returns:
            True if the instance is a dataclass
        """
        return is_dataclass(instance)

    def extract_from_annotation(self, annotation: Any) -> GenericInfo:
        """Extract generic information from a dataclass type annotation.
        
        Args:
            annotation: The dataclass type annotation
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> @dataclass
            ... class Box(Generic[A]):
            ...     item: A
            >>> extractor = DataclassExtractor()
            >>> info = extractor.extract_from_annotation(Box[int])
            >>> info.origin  # Box
            >>> info.concrete_args[0].origin  # int
        """
        origin = get_origin(annotation) or annotation
        args = get_args(annotation)

        if not is_dataclass(origin):
            return GenericInfo()

        concrete_args = [get_generic_info(arg) for arg in args]

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def extract_from_instance(self, instance: Any) -> GenericInfo:
        """Extract generic information from a dataclass instance.
        
        Args:
            instance: The dataclass instance
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> box = Box[int](item=42)
            >>> extractor = DataclassExtractor()
            >>> info = extractor.extract_from_instance(box)
            >>> info.origin  # Box
            >>> info.concrete_args[0].origin  # int (if __orig_class__ exists)
        """
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

        return GenericInfo(origin=origin, concrete_args=concrete_args)

    def _get_original_type_parameters(self, dataclass_class: Any) -> List[TypeVar]:
        """Get the original TypeVar parameters from a dataclass class definition.
        
        Extracts TypeVars from __orig_bases__ by looking for Generic[...] inheritance.
        
        Args:
            dataclass_class: The dataclass class
            
        Returns:
            List of TypeVars defined in the class
        """
        for base in getattr(dataclass_class, "__orig_bases__", []):
            if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                origin = get_origin(base)
                if origin and hasattr(origin, "__name__") and "Generic" in str(origin):
                    args = get_args(base)
                    return [arg for arg in args if isinstance(arg, TypeVar)]
        return []

    def _build_inheritance_aware_substitution(
        self, annotation_info: GenericInfo, instance: Any
    ) -> Dict[TypeVar, GenericInfo]:
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
        if not hasattr(instance_origin, "__orig_bases__"):
            return typevar_map

        # For each parent class, map its TypeVars to the annotation's TypeVars
        for base in instance_origin.__orig_bases__:
            base_origin = get_origin(base) or base
            if base_origin == annotation_class or not is_dataclass(base_origin):
                continue  # Skip self and non-dataclasses

            # Found a parent dataclass! e.g., base = HasA[B, A] where B, A are from HasB
            base_args = get_args(base)  # [B, A] (TypeVars from instance class)
            instance_params = getattr(
                instance_origin, "__parameters__", ()
            )  # (A, B) from HasB

            # Get the parent class's original TypeVars
            parent_class_params = self._get_original_type_parameters(
                base_origin
            )  # [A, B] from HasA

            # Map each parent TypeVar to the annotation's TypeVar
            # parent_class_params[i] (HasA's TypeVar) appears as base_args[i] (HasB's TypeVar reference)
            # We need to find where base_args[i] appears in instance_params and use annotation_info.concrete_args

            for i, parent_tv in enumerate(parent_class_params):
                if i < len(base_args):
                    base_arg = base_args[
                        i
                    ]  # This is a TypeVar from instance class (e.g., B from HasB)

                    if isinstance(base_arg, TypeVar):
                        # Find where this TypeVar appears in the instance's parameters
                        try:
                            param_idx = list(instance_params).index(base_arg)
                            # Map parent TypeVar to annotation's corresponding TypeVar
                            if param_idx < len(annotation_info.concrete_args):
                                typevar_map[parent_tv] = annotation_info.concrete_args[
                                    param_idx
                                ]
                        except ValueError:
                            pass
                    else:
                        # base_arg is a concrete type
                        typevar_map[parent_tv] = get_generic_info(base_arg)

        return typevar_map

    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
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

        # Build TypeVar substitution map
        # Check if this class has parent classes with generic parameters
        has_generic_parents = False
        if hasattr(annotation_class, "__orig_bases__"):
            for base in annotation_class.__orig_bases__:
                base_origin = get_origin(base) or base
                if base_origin != annotation_class and is_dataclass(base_origin):
                    # Has a dataclass parent
                    has_generic_parents = True
                    break

        if has_generic_parents and hasattr(instance, "__orig_class__"):
            # This class inherits from generic dataclasses - need to track inheritance substitution
            # Example: HasB[C, D] where HasB inherits HasA[B, A] (swapped)
            typevar_map = self._build_inheritance_aware_substitution(
                annotation_info, instance
            )
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
            field_hints = get_type_hints(
                annotation_info.origin, globalns=globalns, localns=localns
            )
        except (TypeError, NameError, AttributeError):
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
                field_generic_info = self._substitute_typevars_in_generic_info(
                    field_generic_info, typevar_map
                )

            pairs.append((field_generic_info, field_value))

        return pairs


class GenericTypeUtils:
    """Unified interface for extracting generic type information.
    
    GenericTypeUtils provides a single entry point for extracting structural
    type information from different generic type systems. It delegates to
    specialized extractors based on the type being processed.
    
    The class maintains a registry of extractors and provides methods to:
    - Extract GenericInfo from annotations and instances
    - Generate annotation-value pairs for type inference
    - Handle TypeVars and Union types
    
    Extractors are tried in order: BuiltinExtractor, PydanticExtractor,
    DataclassExtractor, UnionExtractor.
    """

    def __init__(self):
        """Initialize the utility with specialized extractors."""
        self.extractors = [
            BuiltinExtractor(),
            PydanticExtractor(),
            DataclassExtractor(),
            UnionExtractor(),
        ]

    def get_generic_info(self, annotation: Any) -> GenericInfo:
        """Extract generic type information from an annotation.
        
        Args:
            annotation: The type annotation to extract from
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> utils = GenericTypeUtils()
            >>> info = utils.get_generic_info(list[int])
            >>> info.origin  # <class 'list'>
            >>> info.concrete_args[0].origin  # <class 'int'>
        """
        if isinstance(annotation, TypeVar):
            return GenericInfo(origin=annotation)

        for extractor in self.extractors:
            if extractor.can_handle_annotation(annotation):
                return extractor.extract_from_annotation(annotation)

        # Fallback for non-generic types
        return GenericInfo(origin=annotation)

    def get_instance_generic_info(self, instance: Any) -> GenericInfo:
        """Extract generic type information from an instance.
        
        Args:
            instance: The runtime instance to extract from
            
        Returns:
            GenericInfo representing the structural type information
            
        Example:
            >>> utils = GenericTypeUtils()
            >>> info = utils.get_instance_generic_info([1, 2, 3])
            >>> info.origin  # <class 'list'>
        """
        for extractor in self.extractors:
            if extractor.can_handle_instance(instance):
                return extractor.extract_from_instance(instance)

        # Fallback for non-generic instances
        return GenericInfo(origin=type(instance))

    def get_annotation_value_pairs(
        self, annotation: Any, instance: Any
    ) -> List[Tuple[GenericInfo, Any]]:
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
        if hasattr(instance, "__dict__") and annotation_info.concrete_args:
            pairs = []
            first_typevar_info = annotation_info.concrete_args[0]
            for key, value in instance.__dict__.items():
                # Skip special attributes that shouldn't be used for type inference
                if not key.startswith("__"):
                    pairs.append((first_typevar_info, value))
            return pairs

        return []


def create_union_if_needed(types_set: set) -> Any:
    """Create a Union type if needed, or return single type.

    Uses modern union syntax (int | str) for Python 3.10+ compatibility.
    Falls back to typing.Union for edge cases where the | operator doesn't work.
    
    Args:
        types_set: Set of types to union together
        
    Returns:
        Single type if only one type, Union type if multiple types
        
    Example:
        >>> create_union_if_needed({int})
        <class 'int'>
        >>> create_union_if_needed({int, str})
        int | str
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


def create_generic_info_union_if_needed(
    generic_info_set: set[GenericInfo],
) -> GenericInfo:
    """Create a GenericInfo Union type if needed, or return single GenericInfo.

    Works entirely within GenericInfo objects without resolving to actual types.
    This is useful for type inference where we want to maintain the structural
    representation without materializing the final type.
    
    Args:
        generic_info_set: Set of GenericInfo objects to union together
        
    Returns:
        Single GenericInfo if only one, Union GenericInfo if multiple
        
    Example:
        >>> info1 = GenericInfo(origin=int)
        >>> info2 = GenericInfo(origin=str)
        >>> union_info = create_generic_info_union_if_needed({info1, info2})
        >>> union_info.origin  # Union
        >>> len(union_info.concrete_args)  # 2
    """
    if len(generic_info_set) == 1:
        return list(generic_info_set)[0]
    elif len(generic_info_set) > 1:
        # Create a Union GenericInfo with all the GenericInfo objects as concrete_args
        return GenericInfo(origin=Union, concrete_args=list(generic_info_set))
    else:
        # Empty set - return GenericInfo for NoneType
        return GenericInfo(origin=type(None))


# Global instance for convenience
generic_utils = GenericTypeUtils()


# Convenience functions that mirror the class methods
def get_generic_info(annotation: Any) -> GenericInfo:
    """Extract generic type information from an annotation.
    
    Convenience function that delegates to the global GenericTypeUtils instance.
    
    Args:
        annotation: The type annotation to extract from
        
    Returns:
        GenericInfo representing the structural type information
        
    Example:
        >>> info = get_generic_info(list[int])
        >>> info.origin  # <class 'list'>
        >>> info.concrete_args[0].origin  # <class 'int'>
    """
    return generic_utils.get_generic_info(annotation)


def get_instance_generic_info(instance: Any) -> GenericInfo:
    """Extract generic type information from an instance.
    
    Convenience function that delegates to the global GenericTypeUtils instance.
    
    Args:
        instance: The runtime instance to extract from
        
    Returns:
        GenericInfo representing the structural type information
        
    Example:
        >>> info = get_instance_generic_info([1, 2, 3])
        >>> info.origin  # <class 'list'>
    """
    return generic_utils.get_instance_generic_info(instance)


def get_annotation_value_pairs(
    annotation: Any, instance: Any
) -> List[Tuple["GenericInfo", Any]]:
    """Extract (GenericInfo, value) pairs for type inference.
    
    Convenience function that delegates to the global GenericTypeUtils instance.
    
    Args:
        annotation: The type annotation
        instance: The runtime instance
        
    Returns:
        List of (GenericInfo, value) pairs for type inference
        
    Example:
        >>> pairs = get_annotation_value_pairs(list[A], [1, 2, 3])
        >>> len(pairs)  # 3
        >>> pairs[0][0].origin  # A (the TypeVar)
        >>> pairs[0][1]  # 1 (the value)
    """
    return generic_utils.get_annotation_value_pairs(annotation, instance)
