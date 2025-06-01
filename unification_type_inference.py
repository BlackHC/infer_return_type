"""
Unification-based type inference system for generic function return types.

This implements a formal unification algorithm that can handle:
1. Complex nested generic structures
2. TypeVar bounds and constraints  
3. Variance (covariance/contravariance)
4. Union formation when conflicts arise
5. Common interface for different generic type systems

The key insight is to treat type inference as a constraint satisfaction problem
where we unify annotation structures with concrete value types.
"""

import inspect
import typing
import types
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, get_origin, get_args
from dataclasses import fields, is_dataclass
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict


class UnificationError(Exception):
    """Raised when unification fails."""
    pass


class TypeInferenceError(Exception):
    """Raised when type inference fails."""
    pass


class Variance(Enum):
    """Type variance for generic parameters."""
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    INVARIANT = "invariant"


class TypeExtractor(ABC):
    """Abstract interface for extracting type information from different generic type systems."""
    
    @abstractmethod
    def can_handle(self, annotation: Any, instance: Any) -> bool:
        """Check if this extractor can handle the given annotation/instance pair."""
        pass
    
    @abstractmethod
    def extract_type_params(self, annotation: Any) -> List[TypeVar]:
        """Extract TypeVar parameters from the annotation."""
        pass
    
    @abstractmethod
    def extract_concrete_types(self, instance: Any) -> List[type]:
        """Extract concrete types from the instance."""
        pass
    
    @abstractmethod
    def get_variance(self, annotation: Any, param_index: int) -> Variance:
        """Get variance for a specific type parameter."""
        pass


class PydanticExtractor(TypeExtractor):
    """Type extractor for Pydantic generic models."""
    
    def can_handle(self, annotation: Any, instance: Any) -> bool:
        return (hasattr(annotation, '__pydantic_generic_metadata__') or 
                hasattr(instance, '__pydantic_generic_metadata__'))
    
    def extract_type_params(self, annotation: Any) -> List[TypeVar]:
        if hasattr(annotation, '__pydantic_generic_metadata__'):
            metadata = annotation.__pydantic_generic_metadata__
            params = metadata.get('parameters', ())
            return [p for p in params if isinstance(p, TypeVar)]
        return []
    
    def extract_concrete_types(self, instance: Any) -> List[type]:
        if hasattr(instance, '__pydantic_generic_metadata__'):
            metadata = instance.__pydantic_generic_metadata__
            return list(metadata.get('args', ()))
        return []
    
    def get_variance(self, annotation: Any, param_index: int) -> Variance:
        # Pydantic models are generally invariant
        return Variance.INVARIANT


class DataclassExtractor(TypeExtractor):
    """Type extractor for dataclass generic types."""
    
    def can_handle(self, annotation: Any, instance: Any) -> bool:
        return (is_dataclass(annotation) or 
                (hasattr(instance, '__orig_class__') and is_dataclass(instance)))
    
    def extract_type_params(self, annotation: Any) -> List[TypeVar]:
        # For dataclasses, TypeVars are preserved in annotation args
        args = get_args(annotation)
        return [arg for arg in args if isinstance(arg, TypeVar)]
    
    def extract_concrete_types(self, instance: Any) -> List[type]:
        if hasattr(instance, '__orig_class__'):
            return list(get_args(instance.__orig_class__))
        return []
    
    def get_variance(self, annotation: Any, param_index: int) -> Variance:
        # Dataclasses are generally invariant
        return Variance.INVARIANT


class BuiltinExtractor(TypeExtractor):
    """Type extractor for built-in generic types like List, Dict, etc."""
    
    def can_handle(self, annotation: Any, instance: Any) -> bool:
        origin = get_origin(annotation)
        return origin in (list, dict, tuple, set, List, Dict, Tuple, Set)
    
    def extract_type_params(self, annotation: Any) -> List[TypeVar]:
        args = get_args(annotation)
        return [arg for arg in args if isinstance(arg, TypeVar)]
    
    def extract_concrete_types(self, instance: Any) -> List[type]:
        # For built-ins, we infer from the instance content
        if isinstance(instance, list) and instance:
            element_types = {type(item) for item in instance}
            return [_create_union_type(element_types)]
        elif isinstance(instance, dict) and instance:
            key_types = {type(k) for k in instance.keys()}
            value_types = {type(v) for v in instance.values()}
            return [_create_union_type(key_types), _create_union_type(value_types)]
        elif isinstance(instance, tuple) and instance:
            return [type(item) for item in instance]
        elif isinstance(instance, set) and instance:
            element_types = {type(item) for item in instance}
            return [_create_union_type(element_types)]
        return []
    
    def get_variance(self, annotation: Any, param_index: int) -> Variance:
        origin = get_origin(annotation)
        if origin in (list, List, set, Set):
            return Variance.COVARIANT  # List[A] is covariant in A
        elif origin in (dict, Dict):
            if param_index == 0:
                return Variance.INVARIANT  # Dict keys are invariant
            else:
                return Variance.COVARIANT  # Dict values are covariant
        elif origin in (tuple, Tuple):
            return Variance.COVARIANT  # Tuple is covariant
        return Variance.INVARIANT


class Constraint:
    """Represents a type constraint between a TypeVar and a concrete type."""
    
    def __init__(self, typevar: TypeVar, concrete_type: type, variance: Variance = Variance.INVARIANT, is_override: bool = False):
        self.typevar = typevar
        self.concrete_type = concrete_type
        self.variance = variance
        self.is_override = is_override
    
    def __str__(self):
        override_str = " (override)" if self.is_override else ""
        return f"{self.typevar} ~ {self.concrete_type} ({self.variance.value}){override_str}"
    
    def __repr__(self):
        return self.__str__()


class Substitution:
    """Represents a substitution of TypeVars to concrete types."""
    
    def __init__(self):
        self.bindings: Dict[TypeVar, type] = {}
    
    def bind(self, typevar: TypeVar, concrete_type: type):
        """Bind a TypeVar to a concrete type."""
        self.bindings[typevar] = concrete_type
    
    def get(self, typevar: TypeVar) -> Optional[type]:
        """Get the binding for a TypeVar."""
        return self.bindings.get(typevar)
    
    def apply(self, annotation: Any) -> Any:
        """Apply this substitution to an annotation."""
        return _substitute_typevars(annotation, self.bindings)
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another."""
        result = Substitution()
        # Apply other to our bindings first
        for tv, typ in self.bindings.items():
            result.bind(tv, other.apply(typ) if isinstance(typ, TypeVar) else typ)
        # Add other's bindings
        for tv, typ in other.bindings.items():
            if tv not in result.bindings:
                result.bind(tv, typ)
        return result
    
    def __str__(self):
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.bindings.items()) + "}"


class UnificationEngine:
    """Core unification engine for type inference."""
    
    def __init__(self):
        self.extractors = [
            PydanticExtractor(),
            DataclassExtractor(), 
            BuiltinExtractor()
        ]
    
    def unify_annotation_with_value(
        self, 
        annotation: Any, 
        value: Any,
        constraints: List[Constraint] = None
    ) -> Substitution:
        """
        Unify an annotation with a concrete value to produce TypeVar bindings.
        
        This is the main entry point for type inference.
        """
        if constraints is None:
            constraints = []
        
        # Collect constraints from the annotation/value pair
        self._collect_constraints(annotation, value, constraints)
        
        # Solve the constraint system
        return self._solve_constraints(constraints)
    
    def _collect_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Recursively collect type constraints from annotation/value pairs."""
        
        # Base case: Direct TypeVar
        if isinstance(annotation, TypeVar):
            concrete_type = _infer_type_from_value(value)
            constraints.append(Constraint(annotation, concrete_type))
            return
        
        # Handle Union types
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
            self._handle_union_constraints(annotation, value, constraints)
            return
        
        # Handle Optional (Union[T, None])
        if origin is Union and len(args) == 2 and type(None) in args:
            if value is not None:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                self._collect_constraints(non_none_type, value, constraints)
            return
        
        # Handle generic containers
        if origin in (list, List):
            self._handle_list_constraints(annotation, value, constraints)
        elif origin in (dict, Dict):
            self._handle_dict_constraints(annotation, value, constraints)
        elif origin in (tuple, Tuple):
            self._handle_tuple_constraints(annotation, value, constraints)
        elif origin in (set, Set):
            self._handle_set_constraints(annotation, value, constraints)
        else:
            # Handle custom generic types
            self._handle_custom_generic_constraints(annotation, value, constraints)
    
    def _handle_union_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Union type constraints by trying each alternative."""
        args = get_args(annotation)
        
        # Try each union alternative
        best_constraints = None
        best_score = -1
        
        for alternative in args:
            try:
                temp_constraints = []
                self._collect_constraints(alternative, value, temp_constraints)
                
                # Score this alternative (prefer more specific constraints)
                score = len(temp_constraints)
                if score > best_score:
                    best_score = score
                    best_constraints = temp_constraints
            except (UnificationError, TypeError):
                continue
        
        if best_constraints is not None:
            constraints.extend(best_constraints)
        else:
            raise UnificationError(f"Value {value} doesn't match any alternative in {annotation}")
    
    def _handle_list_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle List[T] constraints."""
        if not isinstance(value, list):
            raise UnificationError(f"Expected list, got {type(value)}")
        
        args = get_args(annotation)
        if len(args) == 1:
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                # Collect types from all elements
                if value:
                    element_types = {type(item) for item in value}
                    union_type = _create_union_type(element_types)
                    constraints.append(Constraint(element_annotation, union_type, Variance.COVARIANT))
            else:
                # Recursively handle each element
                for item in value:
                    self._collect_constraints(element_annotation, item, constraints)
    
    def _handle_dict_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Dict[K, V] constraints."""
        if not isinstance(value, dict):
            raise UnificationError(f"Expected dict, got {type(value)}")
        
        args = get_args(annotation)
        if len(args) == 2:
            key_annotation, value_annotation = args
            
            if isinstance(key_annotation, TypeVar) and value:
                key_types = {type(k) for k in value.keys()}
                union_type = _create_union_type(key_types)
                constraints.append(Constraint(key_annotation, union_type, Variance.INVARIANT))
            
            if isinstance(value_annotation, TypeVar) and value:
                value_types = {type(v) for v in value.values()}
                union_type = _create_union_type(value_types)
                constraints.append(Constraint(value_annotation, union_type, Variance.COVARIANT))
            
            # Recursively handle non-TypeVar annotations
            if not isinstance(key_annotation, TypeVar):
                for key in value.keys():
                    self._collect_constraints(key_annotation, key, constraints)
            
            if not isinstance(value_annotation, TypeVar):
                for val in value.values():
                    self._collect_constraints(value_annotation, val, constraints)
    
    def _handle_tuple_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Tuple constraints."""
        if not isinstance(value, tuple):
            raise UnificationError(f"Expected tuple, got {type(value)}")
        
        args = get_args(annotation)
        
        if len(args) == 2 and args[1] is ...:
            # Variable length tuple: Tuple[T, ...]
            element_annotation = args[0]
            if isinstance(element_annotation, TypeVar) and value:
                element_types = {type(item) for item in value}
                union_type = _create_union_type(element_types)
                constraints.append(Constraint(element_annotation, union_type, Variance.COVARIANT))
            elif not isinstance(element_annotation, TypeVar):
                for item in value:
                    self._collect_constraints(element_annotation, item, constraints)
        else:
            # Fixed length tuple: Tuple[T1, T2, ...]
            for i, item in enumerate(value):
                if i < len(args):
                    self._collect_constraints(args[i], item, constraints)
    
    def _handle_set_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Set[T] constraints."""
        if not isinstance(value, set):
            raise UnificationError(f"Expected set, got {type(value)}")
        
        args = get_args(annotation)
        if len(args) == 1:
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                if value:
                    element_types = {type(item) for item in value}
                    union_type = _create_union_type(element_types)
                    constraints.append(Constraint(element_annotation, union_type, Variance.COVARIANT))
            else:
                for item in value:
                    self._collect_constraints(element_annotation, item, constraints)
    
    def _handle_custom_generic_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle custom generic types using extractors."""
        
        for extractor in self.extractors:
            if extractor.can_handle(annotation, value):
                try:
                    type_params = extractor.extract_type_params(annotation)
                    concrete_types = extractor.extract_concrete_types(value)
                    
                    if len(type_params) == len(concrete_types):
                        for i, (param, concrete) in enumerate(zip(type_params, concrete_types)):
                            variance = extractor.get_variance(annotation, i)
                            constraints.append(Constraint(param, concrete, variance))
                        return
                    
                    # Fallback: try to extract from nested structure
                    self._extract_from_nested_structure(annotation, value, constraints)
                    return
                except Exception:
                    continue
        
        # No extractor could handle it - try fallback methods
        self._extract_from_nested_structure(annotation, value, constraints)
    
    def _extract_from_nested_structure(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Fallback method to extract constraints from nested structures."""
        
        # Try to align annotation structure with value structure
        ann_origin = get_origin(annotation)
        ann_args = get_args(annotation)
        
        if hasattr(value, '__orig_class__'):
            value_type = value.__orig_class__
            value_args = get_args(value_type)
            
            if ann_args and value_args and len(ann_args) == len(value_args):
                for ann_arg, val_arg in zip(ann_args, value_args):
                    if isinstance(ann_arg, TypeVar):
                        constraints.append(Constraint(ann_arg, val_arg))
    
    def _solve_constraints(self, constraints: List[Constraint]) -> Substitution:
        """Solve the constraint system to produce a substitution."""
        
        substitution = Substitution()
        
        # Group constraints by TypeVar
        constraint_groups = defaultdict(list)
        for constraint in constraints:
            constraint_groups[constraint.typevar].append(constraint)
        
        # Solve each TypeVar
        for typevar, typevar_constraints in constraint_groups.items():
            resolved_type = self._resolve_typevar_constraints(typevar, typevar_constraints)
            substitution.bind(typevar, resolved_type)
        
        return substitution
    
    def _resolve_typevar_constraints(self, typevar: TypeVar, constraints: List[Constraint]) -> type:
        """Resolve constraints for a single TypeVar."""
        
        if len(constraints) == 1:
            constraint = constraints[0]
            return self._check_typevar_bounds(typevar, constraint.concrete_type)
        
        # Check if we have any override constraints
        override_constraints = [c for c in constraints if c.is_override]
        non_override_constraints = [c for c in constraints if not c.is_override]
        
        # If we have override constraints, they take precedence
        if override_constraints:
            if len(override_constraints) == 1:
                # Single override - use it
                return self._check_typevar_bounds(typevar, override_constraints[0].concrete_type)
            else:
                # Multiple overrides - they must be consistent
                override_types = [c.concrete_type for c in override_constraints]
                if len(set(override_types)) == 1:
                    return self._check_typevar_bounds(typevar, override_types[0])
                else:
                    raise UnificationError(f"Conflicting override constraints for {typevar}: {override_constraints}")
        
        # No overrides - handle normally
        concrete_types = [c.concrete_type for c in constraints]
        
        # Check if all constraints are the same
        if len(set(concrete_types)) == 1:
            return self._check_typevar_bounds(typevar, concrete_types[0])
        
        # Different constraints - create union based on variance
        variances = [c.variance for c in constraints]
        
        if all(v == Variance.COVARIANT for v in variances):
            # All covariant - create union
            return self._check_typevar_bounds(typevar, _create_union_type(set(concrete_types)))
        elif all(v == Variance.CONTRAVARIANT for v in variances):
            # All contravariant - find common supertype (simplified)
            return self._check_typevar_bounds(typevar, _find_common_supertype(concrete_types))
        else:
            # Mixed variance or invariant - require exact match or fail
            raise UnificationError(f"Conflicting constraints for {typevar}: {constraints}")
    
    def _check_typevar_bounds(self, typevar: TypeVar, concrete_type: type) -> type:
        """Check if concrete type satisfies TypeVar bounds and constraints."""
        
        # Check explicit constraints (e.g., TypeVar('T', int, str))
        if typevar.__constraints__:
            if concrete_type not in typevar.__constraints__:
                # For Union types, check if all components are in constraints
                origin = get_origin(concrete_type)
                if origin is Union:
                    union_args = get_args(concrete_type)
                    if not all(arg in typevar.__constraints__ for arg in union_args):
                        raise UnificationError(
                            f"Type {concrete_type} violates constraints {typevar.__constraints__} for {typevar}"
                        )
                else:
                    raise UnificationError(
                        f"Type {concrete_type} violates constraints {typevar.__constraints__} for {typevar}"
                    )
        
        # Check bound (e.g., TypeVar('T', bound=int))
        if typevar.__bound__:
            if not _is_subtype(concrete_type, typevar.__bound__):
                raise UnificationError(
                    f"Type {concrete_type} doesn't satisfy bound {typevar.__bound__} for {typevar}"
                )
        
        return concrete_type


def _create_union_type(types_set: Set[type]) -> type:
    """Create a Union type from a set of types."""
    if len(types_set) == 1:
        return list(types_set)[0]
    else:
        try:
            # Use modern union syntax for Python 3.10+
            result = types_set.pop()
            for elem_type in types_set:
                result = result | elem_type
            return result
        except TypeError:
            # Fallback for older Python versions
            return Union[tuple(types_set)]


def _find_common_supertype(types_list: List[type]) -> type:
    """Find the most specific common supertype of a list of types."""
    if not types_list:
        return type(None)
    
    if len(types_list) == 1:
        return types_list[0]
    
    # Simplified implementation - in practice would need more sophisticated MRO analysis
    # For now, just return Union
    return _create_union_type(set(types_list))


def _is_subtype(subtype: type, supertype: type) -> bool:
    """Check if subtype is a subtype of supertype."""
    try:
        return issubclass(subtype, supertype)
    except TypeError:
        # Handle cases where subtype might not be a class
        return False


def _infer_type_from_value(value: Any) -> type:
    """Infer the most specific type from a value."""
    if value is None:
        return type(None)
    
    base_type = type(value)
    
    # For collections, try to infer element types
    if isinstance(value, list) and value:
        element_types = {type(item) for item in value}
        if len(element_types) == 1:
            element_type = list(element_types)[0]
            return list[element_type]
        else:
            return list[_create_union_type(element_types)]
    elif isinstance(value, dict) and value:
        key_types = {type(k) for k in value.keys()}
        value_types = {type(v) for v in value.values()}
        
        key_type = list(key_types)[0] if len(key_types) == 1 else _create_union_type(key_types)
        value_type = list(value_types)[0] if len(value_types) == 1 else _create_union_type(value_types)
        
        return dict[key_type, value_type]
    elif isinstance(value, tuple):
        element_types = tuple(type(item) for item in value)
        return tuple[element_types]
    elif isinstance(value, set) and value:
        element_types = {type(item) for item in value}
        if len(element_types) == 1:
            element_type = list(element_types)[0]
            return set[element_type]
        else:
            return set[_create_union_type(element_types)]
    
    return base_type


def _substitute_typevars(annotation: Any, bindings: Dict[TypeVar, type]) -> Any:
    """Substitute TypeVars in an annotation with their bindings."""
    
    if isinstance(annotation, TypeVar):
        if annotation in bindings:
            return bindings[annotation]
        else:
            raise UnificationError(f"Unbound TypeVar: {annotation}")
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if not origin or not args:
        return annotation
    
    # Recursively substitute in type arguments
    substituted_args = []
    for arg in args:
        substituted_args.append(_substitute_typevars(arg, bindings))
    
    # Reconstruct the type
    if origin in (list, List):
        return list[substituted_args[0]]
    elif origin in (dict, Dict):
        return dict[substituted_args[0], substituted_args[1]]
    elif origin in (tuple, Tuple):
        return tuple[tuple(substituted_args)]
    elif origin in (set, Set):
        return set[substituted_args[0]]
    elif origin is Union:
        if len(substituted_args) == 1:
            return substituted_args[0]
        return _create_union_type(set(substituted_args))
    else:
        # For other generic types, try to reconstruct
        try:
            return origin[tuple(substituted_args)]
        except Exception:
            return annotation


def infer_return_type_unified(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs,
) -> type:
    """
    Infer the concrete return type using unification algorithm.
    
    This is the main entry point that replaces the original infer_return_type.
    """
    
    if type_overrides is None:
        type_overrides = {}
    
    # Get function signature and return annotation
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    
    if return_annotation is inspect.Signature.empty:
        raise ValueError("Function must have return type annotation")
    
    # Create unification engine
    engine = UnificationEngine()
    
    # Collect all constraints from function parameters
    all_constraints = []
    
    # Process positional arguments
    param_names = list(sig.parameters.keys())
    for i, arg in enumerate(args):
        if i < len(param_names):
            param = sig.parameters[param_names[i]]
            if param.annotation != inspect.Parameter.empty:
                engine._collect_constraints(param.annotation, arg, all_constraints)
    
    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                engine._collect_constraints(param.annotation, value, all_constraints)
    
    # Add type overrides as constraints
    for typevar, override_type in type_overrides.items():
        all_constraints.append(Constraint(typevar, override_type, is_override=True))
    
    # Solve constraints to get substitution
    try:
        substitution = engine._solve_constraints(all_constraints)
    except UnificationError as e:
        raise TypeInferenceError(str(e))
    
    # Apply substitution to return annotation
    return substitution.apply(return_annotation) 