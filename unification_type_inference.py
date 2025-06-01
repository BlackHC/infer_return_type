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
        
        try:
            self._collect_constraints_internal(annotation, value, constraints)
        except UnificationError as e:
            # Convert to TypeInferenceError for consistency
            raise TypeInferenceError(str(e))
    
    def _collect_constraints_internal(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Internal constraint collection that can raise UnificationError."""
        
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
                self._collect_constraints_internal(non_none_type, value, constraints)
            else:
                # Value is None - we still need to handle the TypeVar
                non_none_type = args[0] if args[1] is type(None) else args[1]
                if isinstance(non_none_type, TypeVar):
                    # For Optional[A] with None value, we can't infer A but shouldn't fail
                    pass
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
                self._collect_constraints_internal(alternative, value, temp_constraints)
                
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
                # If empty list, don't add constraint - will be handled later
            else:
                # Handle Union inside List specially
                origin = get_origin(element_annotation)
                if origin is Union:
                    # For List[Union[A, B]] with values [int, str], we need to collect constraints differently
                    union_args = get_args(element_annotation)
                    for item in value:
                        # Try to match item against union alternatives
                        self._match_value_to_union_alternatives(item, union_args, constraints)
                else:
                    # Recursively handle each element
                    for item in value:
                        self._collect_constraints_internal(element_annotation, item, constraints)
    
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
                    self._collect_constraints_internal(key_annotation, key, constraints)
            
            if not isinstance(value_annotation, TypeVar):
                for val in value.values():
                    self._collect_constraints_internal(value_annotation, val, constraints)
    
    def _handle_tuple_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Tuple constraints - process sub-constraints in order for better context."""
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
                    self._collect_constraints_internal(element_annotation, item, constraints)
        else:
            # Fixed length tuple: Process in order for better constraint propagation
            for i, item in enumerate(value):
                if i < len(args):
                    # Process each tuple element, building up constraint context
                    self._collect_constraints_internal(args[i], item, constraints)
    
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
                # Check if this is a Union type
                origin = get_origin(element_annotation)
                if origin is Union:
                    union_args = get_args(element_annotation)
                    for item in value:
                        self._match_value_to_union_alternatives(item, union_args, constraints)
                else:
                    for item in value:
                        self._collect_constraints_internal(element_annotation, item, constraints)
    
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
        
        # First, check if the value has explicit type information
        if hasattr(value, '__orig_class__'):
            value_type = value.__orig_class__
            value_args = get_args(value_type)
            
            if ann_args and value_args and len(ann_args) == len(value_args):
                for ann_arg, val_arg in zip(ann_args, value_args):
                    if isinstance(ann_arg, TypeVar):
                        constraints.append(Constraint(ann_arg, val_arg))
                    else:
                        # Recursively handle nested generic structures
                        self._extract_from_annotation_alignment(ann_arg, val_arg, constraints)
                return
        
        # Try to extract from instance fields for dataclasses and similar
        self._extract_from_instance_fields(annotation, value, constraints)
    
    def _extract_from_annotation_alignment(self, ann_type: Any, val_type: Any, constraints: List[Constraint]):
        """Extract constraints by aligning annotation and value type structures."""
        
        ann_origin = get_origin(ann_type)
        ann_args = get_args(ann_type)
        val_origin = get_origin(val_type)
        val_args = get_args(val_type)
        
        # Special handling for Pydantic generic classes
        # The annotation might be just `Box` but we need to extract TypeVars from class metadata
        if (not ann_args and hasattr(ann_type, '__pydantic_generic_metadata__') and 
            val_args and hasattr(val_type, '__pydantic_generic_metadata__')):
            
            ann_metadata = ann_type.__pydantic_generic_metadata__
            val_metadata = val_type.__pydantic_generic_metadata__
            
            # Get TypeVars from annotation class and concrete types from value type
            ann_typevars = ann_metadata.get('parameters', ())
            val_concrete_types = val_metadata.get('args', ()) or get_args(val_type)
            
            if ann_typevars and val_concrete_types and len(ann_typevars) == len(val_concrete_types):
                for ann_typevar, val_concrete in zip(ann_typevars, val_concrete_types):
                    if isinstance(ann_typevar, TypeVar):
                        constraints.append(Constraint(ann_typevar, val_concrete))
                return
        
        # If both have the same origin and compatible args, align them
        if ann_origin == val_origin and ann_args and val_args and len(ann_args) == len(val_args):
            for i, (ann_arg, val_arg) in enumerate(zip(ann_args, val_args)):
                if isinstance(ann_arg, TypeVar):
                    constraints.append(Constraint(ann_arg, val_arg))
                else:
                    # Recursively handle deeper nesting
                    self._extract_from_annotation_alignment(ann_arg, val_arg, constraints)
        elif isinstance(ann_type, TypeVar):
            # Direct TypeVar binding
            constraints.append(Constraint(ann_type, val_type))
        else:
            # For non-matching cases, try Pydantic extraction
            if (hasattr(ann_type, '__pydantic_generic_metadata__') and 
                hasattr(val_type, '__pydantic_generic_metadata__')):
                self._extract_pydantic_constraints(ann_type, val_type, constraints)
    
    def _extract_pydantic_constraints(self, ann_type: Any, val_type: Any, constraints: List[Constraint]):
        """Extract constraints from Pydantic generic types."""
        ann_metadata = ann_type.__pydantic_generic_metadata__
        val_metadata = val_type.__pydantic_generic_metadata__
        
        # Get TypeVars from annotation class and concrete types from value type
        ann_typevars = ann_metadata.get('parameters', ())
        val_concrete_types = val_metadata.get('args', ())
        
        if ann_typevars and val_concrete_types and len(ann_typevars) == len(val_concrete_types):
            for ann_typevar, val_concrete in zip(ann_typevars, val_concrete_types):
                if isinstance(ann_typevar, TypeVar):
                    constraints.append(Constraint(ann_typevar, val_concrete))
    
    def _extract_from_instance_fields(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Extract type constraints from instance field values."""
        
        # Handle dataclasses
        if is_dataclass(value):
            ann_args = get_args(annotation)
            if ann_args:
                # Try to infer TypeVar bindings from field values
                field_values = []
                for field in fields(value):
                    field_value = getattr(value, field.name)
                    field_values.append((field.name, field_value, type(field_value)))
                
                # For now, simple heuristic: if there's one TypeVar and all fields have same type
                if len(ann_args) == 1 and isinstance(ann_args[0], TypeVar):
                    if field_values:
                        # Collect types from all fields and create union
                        field_types = {fv[2] for fv in field_values}
                        union_type = _create_union_type(field_types)
                        constraints.append(Constraint(ann_args[0], union_type))
        
        # Handle other generic instances by trying to extract from __dict__
        elif hasattr(value, '__dict__') and hasattr(annotation, '__args__'):
            ann_args = get_args(annotation)
            if ann_args and len(ann_args) == 1 and isinstance(ann_args[0], TypeVar):
                # Extract types from instance attributes
                attr_types = {type(v) for v in value.__dict__.values() if v is not None}
                if attr_types:
                    union_type = _create_union_type(attr_types)
                    constraints.append(Constraint(ann_args[0], union_type))
    
    def _solve_constraints(self, constraints: List[Constraint]) -> Substitution:
        """Solve the constraint system to produce a substitution with global context awareness."""
        
        substitution = Substitution()
        
        # Group constraints by TypeVar
        constraint_groups = defaultdict(list)
        for constraint in constraints:
            constraint_groups[constraint.typevar].append(constraint)
        
        # First pass: resolve TypeVars with unambiguous constraints
        resolved_in_first_pass = set()
        
        for typevar, typevar_constraints in constraint_groups.items():
            if self._can_resolve_unambiguously(typevar_constraints):
                resolved_type = self._resolve_typevar_constraints(typevar, typevar_constraints)
                substitution.bind(typevar, resolved_type)
                resolved_in_first_pass.add(typevar)
        
        # Second pass: resolve remaining TypeVars using context from first pass
        for typevar, typevar_constraints in constraint_groups.items():
            if typevar not in resolved_in_first_pass:
                # Try to use context from already resolved TypeVars
                refined_constraints = self._refine_constraints_with_context(
                    typevar_constraints, substitution
                )
                resolved_type = self._resolve_typevar_constraints(typevar, refined_constraints)
                substitution.bind(typevar, resolved_type)
        
        return substitution
    
    def _can_resolve_unambiguously(self, constraints: List[Constraint]) -> bool:
        """Check if constraints can be resolved without ambiguity."""
        if len(constraints) <= 1:
            return True
        
        # If all constraints have the same concrete type, unambiguous
        concrete_types = [c.concrete_type for c in constraints]
        if len(set(concrete_types)) == 1:
            return True
        
        # If we have only invariant constraints with different types, this is ambiguous (conflict)
        variances = [c.variance for c in constraints]
        if all(v == Variance.INVARIANT for v in variances):
            return len(set(concrete_types)) == 1  # Only unambiguous if all same type
        
        # If we have overrides, those are unambiguous
        if any(c.is_override for c in constraints):
            return True
        
        # Covariant constraints can be resolved (by union formation)
        return True
    
    def _refine_constraints_with_context(
        self, 
        constraints: List[Constraint], 
        context: Substitution
    ) -> List[Constraint]:
        """Refine constraints using context from already resolved TypeVars."""
        
        # For now, just return original constraints
        # This is where we could implement more sophisticated constraint propagation
        # For example, if we know A=int and B=str, and we have Set[A|B] with {1, "hello"},
        # we could refine the constraints to be more specific
        
        return constraints
    
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
        
        # Different constraints - analyze variance and context
        variances = [c.variance for c in constraints]
        
        # Handle None values: Instead of ignoring None, include it in the union when appropriate
        none_types = [c for c in constraints if c.concrete_type == type(None)]
        non_none_constraints = [c for c in constraints if c.concrete_type != type(None)]
        
        if none_types and non_none_constraints:
            # Both None and non-None types present - create union including None
            non_none_types = [c.concrete_type for c in non_none_constraints]
            all_types = set(non_none_types) | {type(None)}
            return self._check_typevar_bounds(typevar, _create_union_type(all_types))
        
        # Key insight: distinguish between "forced unions" and "conflicting sources"
        # Forced unions: single container with mixed types (List[A] with mixed elements)
        # Conflicting sources: multiple separate containers claiming different types for same TypeVar
        
        # Check if constraints come from the same "source context" or different sources
        covariant_constraints = [c for c in constraints if c.variance == Variance.COVARIANT]
        invariant_constraints = [c for c in constraints if c.variance == Variance.INVARIANT]
        
        # If we have covariant constraints (like List[A] with mixed elements), form union
        if covariant_constraints and not invariant_constraints:
            return self._check_typevar_bounds(typevar, _create_union_type(set(concrete_types)))
        
        # If we have multiple invariant constraints (like multiple Dict[A,B] or List[A]), this is a conflict
        if len(invariant_constraints) > 1:
            invariant_types = [c.concrete_type for c in invariant_constraints]
            if len(set(invariant_types)) > 1:
                # True conflict - multiple independent sources claiming different types
                raise UnificationError(f"Conflicting type assignments for {typevar}: {invariant_constraints}")
        
        # Mixed variance - default to union formation
        return self._check_typevar_bounds(typevar, _create_union_type(set(concrete_types)))
    
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

    def _match_value_to_union_alternatives(self, value: Any, union_alternatives: Tuple, constraints: List[Constraint]):
        """Match a value against union alternatives and collect constraints."""
        value_type = type(value)
        
        # First, check if the value exactly matches any concrete (non-TypeVar) type in the union
        # This handles cases like Optional[A] where None should match the concrete None type
        for alt in union_alternatives:
            if not isinstance(alt, TypeVar) and alt == value_type:
                # Perfect match with concrete type - no constraints needed
                return
        
        # Check if we already have strong hints about what each TypeVar should be
        # by looking at existing constraints
        existing_bindings = {}
        for constraint in constraints:
            if constraint.typevar in union_alternatives:
                # Only consider "strong" evidence (invariant constraints like from Dict keys/values)
                if constraint.variance == Variance.INVARIANT:
                    if constraint.typevar not in existing_bindings:
                        existing_bindings[constraint.typevar] = set()
                    existing_bindings[constraint.typevar].add(constraint.concrete_type)
        
        # Try to assign this value to the TypeVar that already has evidence for this type
        matched_typevar = None
        
        for alt in union_alternatives:
            if isinstance(alt, TypeVar) and alt in existing_bindings:
                existing_types = existing_bindings[alt]
                if len(existing_types) == 1 and value_type in existing_types:
                    # Perfect match - this TypeVar already has evidence for this exact type
                    matched_typevar = alt
                    break
        
        # If we found a perfect match, use it
        if matched_typevar:
            # Add a covariant constraint since this is coming from a Set/collection
            constraints.append(Constraint(matched_typevar, value_type, Variance.COVARIANT))
            return
        
        # No perfect match found - check if we can rule out some TypeVars
        ruled_out = set()
        for alt in union_alternatives:
            if isinstance(alt, TypeVar) and alt in existing_bindings:
                existing_types = existing_bindings[alt]
                if len(existing_types) == 1 and value_type not in existing_types:
                    # This TypeVar has strong evidence for a different type
                    ruled_out.add(alt)
        
        # Add constraints for remaining candidates
        candidates = [alt for alt in union_alternatives 
                     if isinstance(alt, TypeVar) and alt not in ruled_out]
        
        if candidates:
            # If we have candidates that aren't ruled out, use covariant constraints
            # This allows union formation if needed
            for candidate in candidates:
                constraints.append(Constraint(candidate, value_type, Variance.COVARIANT))
        else:
            # Fallback: add constraints for all TypeVar alternatives with invariant variance
            # This will likely cause conflicts but that's better than losing information
            for alt in union_alternatives:
                if isinstance(alt, TypeVar):
                    constraints.append(Constraint(alt, value_type, Variance.INVARIANT))


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
            # Instead of failing, return the TypeVar as-is - this will be caught later
            return annotation
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if not origin or not args:
        return annotation
    
    # Handle Union types specially - only include bound TypeVars
    if origin is Union:
        substituted_args = []
        
        for arg in args:
            substituted_arg = _substitute_typevars(arg, bindings)
            # Only include the arg if it doesn't contain unbound TypeVars
            if not _has_unbound_typevars(substituted_arg):
                substituted_args.append(substituted_arg)
        
        # If we have at least one bound arg, return the union of bound args
        if substituted_args:
            if len(substituted_args) == 1:
                return substituted_args[0]
            # Use helper function to create Union type
            return _create_union_type(set(substituted_args))
        
        # If no args were bound, return the original annotation (will be caught as unbound)
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
    result = substitution.apply(return_annotation)
    
    # Handle case where TypeVars remain unbound (e.g., empty containers)
    if isinstance(result, TypeVar):
        raise TypeInferenceError(f"Could not infer type for {result} - insufficient type information")
    
    # Check for any remaining unbound TypeVars in complex types
    if _has_unbound_typevars(result):
        raise TypeInferenceError(f"Could not fully infer return type - some TypeVars remain unbound: {result}")
    
    return result


def _has_unbound_typevars(annotation: Any) -> bool:
    """Check if an annotation contains any unbound TypeVars."""
    if isinstance(annotation, TypeVar):
        return True
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if args:
        return any(_has_unbound_typevars(arg) for arg in args)
    
    return False 