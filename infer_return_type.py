import inspect
import typing
import types
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, get_origin, get_args
from dataclasses import fields, is_dataclass


class TypeInferenceError(Exception):
    """Raised when type inference fails."""
    pass


def _get_concrete_type_for_typevar_binding(value: Any) -> type:
    """Extract only the concrete base type from a value for TypeVar binding."""
    if value is None:
        return type(None)
    
    # For TypeVar binding, we only care about the concrete base type
    base_type = type(value)
    
    # Handle special cases where we can infer more specific information
    if isinstance(value, (list, dict, tuple, set)) and value:
        if isinstance(value, list):
            # For non-empty lists, try to infer element type if homogeneous
            element_types = {type(item) for item in value}
            if len(element_types) == 1:
                element_type = list(element_types)[0]
                return list[element_type]
            else:
                # Mixed types - use Union
                return list[Union[tuple(element_types)]]
        elif isinstance(value, dict):
            # For non-empty dicts, infer key and value types if homogeneous
            key_types = {type(k) for k in value.keys()}
            value_types = {type(v) for v in value.values()}
            
            key_type = list(key_types)[0] if len(key_types) == 1 else Union[tuple(key_types)]
            value_type = list(value_types)[0] if len(value_types) == 1 else Union[tuple(value_types)]
            
            return dict[key_type, value_type]
        elif isinstance(value, tuple):
            # For tuples, infer element types
            element_types = tuple(type(item) for item in value)
            return tuple[element_types]
        elif isinstance(value, set):
            # For non-empty sets, try to infer element type if homogeneous
            element_types = {type(item) for item in value}
            if len(element_types) == 1:
                element_type = list(element_types)[0]
                return set[element_type]
            else:
                return set[Union[tuple(element_types)]]
    
    return base_type


def _extract_typevar_bindings_from_annotation(
    param_annotation: Any, 
    arg_value: Any, 
    bindings: Dict[TypeVar, type]
) -> None:
    """
    Extract TypeVar bindings by walking the annotation structure and binding 
    TypeVars to concrete types derived from the argument value.
    
    This is annotation-driven: we trust the annotation structure completely
    and only use the value to provide concrete type information for TypeVar binding.
    """
    
    # Base case: Direct TypeVar
    if isinstance(param_annotation, TypeVar):
        concrete_type = _get_concrete_type_for_typevar_binding(arg_value)
        if param_annotation in bindings:
            # Check for conflicts
            if bindings[param_annotation] != concrete_type:
                raise TypeInferenceError(
                    f"Conflicting types for {param_annotation}: "
                    f"{bindings[param_annotation]} vs {concrete_type}"
                )
        else:
            bindings[param_annotation] = concrete_type
        return
    
    # Get annotation structure
    origin = get_origin(param_annotation)
    args = get_args(param_annotation)
    
    if not origin or not args:
        # No generic structure to extract from
        
        # Special case: For Pydantic generics, the annotation might be the raw class
        # without type parameters, but we can still extract TypeVar bindings if
        # the class has generic metadata and the instance has concrete type info
        if (hasattr(param_annotation, '__pydantic_generic_metadata__') and
            hasattr(arg_value, '__pydantic_generic_metadata__')):
            _extract_from_custom_generic(param_annotation, arg_value, param_annotation, (), bindings)
        
        return
    
    # Handle Union types (including Optional[T] which is Union[T, None])
    # Also handle modern union syntax (types.UnionType from Python 3.10+)
    union_type = getattr(types, 'UnionType', None)
    if origin is Union or (union_type and origin is union_type):
        # Handle Optional[T] -> Union[T, None] as a special case
        if len(args) == 2 and type(None) in args:
            if arg_value is not None:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                _extract_typevar_bindings_from_annotation(non_none_type, arg_value, bindings)
            return
        
        # Handle general Union types - try each alternative until one succeeds
        for union_alternative in args:
            try:
                # Create a temporary bindings copy to test this alternative
                temp_bindings = bindings.copy()
                _extract_typevar_bindings_from_annotation(union_alternative, arg_value, temp_bindings)
                # If we succeeded, update the real bindings and return
                bindings.update(temp_bindings)
                return
            except (TypeInferenceError, AttributeError, TypeError) as e:
                # This alternative didn't match, try the next one
                continue
        
        # If no alternative matched, raise an error
        raise TypeInferenceError(
            f"Value {arg_value} of type {type(arg_value)} doesn't match any alternative in Union {args}"
        )
    
    # Handle generic collections - annotation structure drives the extraction
    if origin in (list, List):
        if len(args) == 1 and isinstance(arg_value, list):
            element_annotation = args[0]
            # For each element, try to bind TypeVars using the element annotation
            for item in arg_value:
                _extract_typevar_bindings_from_annotation(element_annotation, item, bindings)
        else:
            # Type mismatch: annotation expects list but value is not a list
            raise TypeInferenceError(f"Expected list but got {type(arg_value)}")
    
    elif origin in (dict, Dict):
        if len(args) == 2 and isinstance(arg_value, dict):
            key_annotation, value_annotation = args
            # Bind TypeVars from keys and values
            for key, value in arg_value.items():
                _extract_typevar_bindings_from_annotation(key_annotation, key, bindings)
                _extract_typevar_bindings_from_annotation(value_annotation, value, bindings)
        else:
            # Type mismatch: annotation expects dict but value is not a dict
            raise TypeInferenceError(f"Expected dict but got {type(arg_value)}")
    
    elif origin in (tuple, Tuple):
        if isinstance(arg_value, tuple):
            # Handle tuple[T, ...] vs tuple[T1, T2, T3]
            if len(args) == 2 and args[1] is ...:
                # Variable length tuple: tuple[T, ...]
                element_annotation = args[0]
                for item in arg_value:
                    _extract_typevar_bindings_from_annotation(element_annotation, item, bindings)
            else:
                # Fixed length tuple: tuple[T1, T2, T3]
                for i, item in enumerate(arg_value):
                    if i < len(args):
                        _extract_typevar_bindings_from_annotation(args[i], item, bindings)
        else:
            # Type mismatch: annotation expects tuple but value is not a tuple
            raise TypeInferenceError(f"Expected tuple but got {type(arg_value)}")
    
    elif origin in (set, Set):
        if len(args) == 1 and isinstance(arg_value, set):
            element_annotation = args[0]
            for item in arg_value:
                _extract_typevar_bindings_from_annotation(element_annotation, item, bindings)
        else:
            # Type mismatch: annotation expects set but value is not a set
            raise TypeInferenceError(f"Expected set but got {type(arg_value)}")
    
    # Handle custom generic types (dataclasses, Pydantic models, etc.)
    else:
        _extract_from_custom_generic(param_annotation, arg_value, origin, args, bindings)


def _extract_typevars_from_annotation(annotation: Any) -> List[TypeVar]:
    """Extract all TypeVars from a nested annotation structure."""
    typevars = []
    
    if isinstance(annotation, TypeVar):
        typevars.append(annotation)
        return typevars
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if args:
        for arg in args:
            typevars.extend(_extract_typevars_from_annotation(arg))
    
    return typevars


def _align_nested_structures(annotation: Any, concrete_type: Any, bindings: Dict[TypeVar, type]) -> None:
    """
    Recursively align annotation structure with concrete type structure to extract TypeVar bindings.
    
    This handles cases like:
    annotation: Wrap[List[Box[A]]]
    concrete_type: Wrap[List[Box[int]]]
    
    Should bind A -> int
    """
    
    # Direct TypeVar case
    if isinstance(annotation, TypeVar):
        if annotation in bindings:
            if bindings[annotation] != concrete_type:
                raise TypeInferenceError(f"Conflicting types for {annotation}")
        else:
            bindings[annotation] = concrete_type
        return
    
    # Get structure of both annotation and concrete type
    ann_origin = get_origin(annotation)
    ann_args = get_args(annotation)
    
    concrete_origin = get_origin(concrete_type)
    concrete_args = get_args(concrete_type)
    
    # If origins don't match, we can't align
    if ann_origin != concrete_origin:
        return
    
    # If no args, nothing to align
    if not ann_args or not concrete_args:
        # Special case: For Pydantic classes, check metadata even when typing system shows no args
        if (hasattr(annotation, '__pydantic_generic_metadata__') and 
            hasattr(concrete_type, '__pydantic_generic_metadata__')):
            
            ann_metadata = annotation.__pydantic_generic_metadata__
            concrete_metadata = concrete_type.__pydantic_generic_metadata__
            
            # Get TypeVars from annotation class and concrete types from concrete class
            ann_typevars = ann_metadata.get('parameters', ())
            concrete_types = concrete_metadata.get('args', ())
            
            if ann_typevars and concrete_types and len(ann_typevars) == len(concrete_types):
                for typevar, concrete_type in zip(ann_typevars, concrete_types):
                    if isinstance(typevar, TypeVar):
                        if typevar in bindings:
                            if bindings[typevar] != concrete_type:
                                raise TypeInferenceError(f"Conflicting types for {typevar}")
                        else:
                            bindings[typevar] = concrete_type
                return
        
        return
    
    # If different number of args, we can't align
    if len(ann_args) != len(concrete_args):
        return
    
    # Recursively align each argument
    for ann_arg, concrete_arg in zip(ann_args, concrete_args):
        _align_nested_structures(ann_arg, concrete_arg, bindings)


def _extract_from_custom_generic(
    param_annotation: Any, 
    instance: Any, 
    origin: type, 
    type_args: tuple, 
    bindings: Dict[TypeVar, type]
) -> None:
    """
    Extract TypeVar bindings from custom generic types like dataclasses and Pydantic models.
    
    This tries multiple strategies to bind TypeVars based on the annotation structure.
    """
    
    # For Pydantic generics, type_args might be empty because get_args() doesn't 
    # preserve TypeVar info in annotations. We need to get it from class metadata.
    effective_type_args = type_args
    
    # Check if this is a Pydantic generic class and we have empty type_args
    if (not type_args and hasattr(origin, '__pydantic_generic_metadata__')):
        class_metadata = origin.__pydantic_generic_metadata__
        if 'parameters' in class_metadata and class_metadata['parameters']:
            effective_type_args = class_metadata['parameters']
    
    # For dataclass annotations, always extract TypeVars from the full annotation structure
    # because get_args() at the top level doesn't give us the nested TypeVars
    if hasattr(instance, '__orig_class__'):
        annotation_typevars = _extract_typevars_from_annotation(param_annotation)
        if annotation_typevars:
            effective_type_args = tuple(annotation_typevars)
    
    # Strategy 1: Try to get concrete type from __orig_class__ (dataclasses)
    if hasattr(instance, '__orig_class__'):
        orig_class = instance.__orig_class__
        orig_args = get_args(orig_class)
        
        if orig_args and len(orig_args) == len(effective_type_args):
            # Check if we actually have TypeVars to bind directly
            has_typevars = any(isinstance(arg, TypeVar) for arg in effective_type_args)
            if has_typevars:
                for param_arg, concrete_type in zip(effective_type_args, orig_args):
                    if isinstance(param_arg, TypeVar):
                        if param_arg in bindings and bindings[param_arg] != concrete_type:
                            raise TypeInferenceError(f"Conflicting types for {param_arg}")
                        bindings[param_arg] = concrete_type
                return
            else:
                # No direct TypeVars, try nested structure alignment
                _align_nested_structures(param_annotation, orig_class, bindings)
                return
    
    # Strategy 2: Try Pydantic generic metadata
    if hasattr(instance, '__pydantic_generic_metadata__'):
        metadata = instance.__pydantic_generic_metadata__
        if 'args' in metadata and metadata['args']:
            concrete_args = metadata['args']
            if len(concrete_args) == len(effective_type_args):
                for param_arg, concrete_type in zip(effective_type_args, concrete_args):
                    if isinstance(param_arg, TypeVar):
                        if param_arg in bindings and bindings[param_arg] != concrete_type:
                            raise TypeInferenceError(f"Conflicting types for {param_arg}")
                        bindings[param_arg] = concrete_type
                return
    
    # Strategy 3: Try to infer from instance fields (dataclasses, etc.)
    if is_dataclass(instance):
        _extract_from_dataclass_fields_fallback(param_annotation, instance, effective_type_args, bindings)
        return
    
    # Strategy 4: Try to infer from instance attributes matching annotation pattern
    _extract_from_instance_attributes(param_annotation, instance, effective_type_args, bindings)


def _extract_from_dataclass_fields_fallback(
    param_annotation: Any, 
    instance: Any, 
    type_args: tuple, 
    bindings: Dict[TypeVar, type]
) -> None:
    """Extract TypeVar bindings from dataclass fields as a fallback strategy."""
    
    dc_fields = fields(instance)
    if not dc_fields:
        return
    
    # Simple heuristic: if there's a single TypeVar, try to infer it from the first field
    typevar_args = [arg for arg in type_args if isinstance(arg, TypeVar)]
    if len(typevar_args) == 1:
        type_var = typevar_args[0]
        first_field_value = getattr(instance, dc_fields[0].name)
        concrete_type = _get_concrete_type_for_typevar_binding(first_field_value)
        
        if type_var in bindings and bindings[type_var] != concrete_type:
            raise TypeInferenceError(f"Conflicting types for {type_var}")
        bindings[type_var] = concrete_type


def _extract_from_instance_attributes(
    param_annotation: Any, 
    instance: Any, 
    type_args: tuple, 
    bindings: Dict[TypeVar, type]
) -> None:
    """
    Try to extract TypeVar bindings by examining instance attributes.
    This is a fallback strategy for custom generic types.
    """
    
    # Try to find attributes that might correspond to TypeVar parameters
    typevar_args = [arg for arg in type_args if isinstance(arg, TypeVar)]
    
    # This is a heuristic - look for common attribute names
    common_attrs = ['value', 'item', 'data', 'content']
    
    for type_var in typevar_args:
        for attr_name in common_attrs:
            if hasattr(instance, attr_name):
                attr_value = getattr(instance, attr_name)
                concrete_type = _get_concrete_type_for_typevar_binding(attr_value)
                
                if type_var in bindings and bindings[type_var] != concrete_type:
                    raise TypeInferenceError(f"Conflicting types for {type_var}")
                bindings[type_var] = concrete_type
                break


def _substitute_type_vars(annotation: Any, bindings: Dict[TypeVar, type]) -> Any:
    """Substitute TypeVars in an annotation with their concrete bindings."""
    
    if isinstance(annotation, TypeVar):
        if annotation in bindings:
            return bindings[annotation]
        else:
            raise TypeInferenceError(f"Unbound TypeVar: {annotation}")
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if not origin or not args:
        return annotation
    
    # Handle Union types specially - they can have partially bound TypeVars
    if origin is Union:
        substituted_args = []
        
        for arg in args:
            try:
                substituted_arg = _substitute_type_vars(arg, bindings)
                substituted_args.append(substituted_arg)
            except TypeInferenceError:
                # Skip unbound TypeVars in unions
                continue
        
        # If we have at least one bound arg, return the union of bound args
        if substituted_args:
            if len(substituted_args) == 1:
                return substituted_args[0]
            # Use modern union syntax for Python 3.10+
            try:
                result = substituted_args[0]
                for arg in substituted_args[1:]:
                    result = result | arg
                return result
            except TypeError:
                # Fallback for older Python versions
                return Union[tuple(substituted_args)]
        
        # If no args were bound, this is an error
        raise TypeInferenceError(f"No TypeVars bound in Union {args}")
    
    # Recursively substitute in type arguments
    substituted_args = []
    for arg in args:
        substituted_args.append(_substitute_type_vars(arg, bindings))
    
    # Handle generic types
    if origin in (list, List):
        return list[substituted_args[0]]
    elif origin in (dict, Dict):
        return dict[substituted_args[0], substituted_args[1]]
    elif origin in (tuple, Tuple):
        return tuple[tuple(substituted_args)]
    elif origin in (set, Set):
        return set[substituted_args[0]]
    else:
        # For other generic types, try to reconstruct
        try:
            return origin[tuple(substituted_args)]
        except Exception:
            # If reconstruction fails, return the original annotation
            # This can happen with complex generic types
            return annotation


def infer_return_type(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs,
) -> type:
    """
    Infer the concrete return type for one call to `fn`.
    
    This is annotation-driven: we use the function's parameter type annotations
    as the primary source of truth and only use argument values to bind TypeVars
    to concrete types.
    """
    
    if type_overrides is None:
        type_overrides = {}
    
    # Get function signature and return annotation
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    
    if return_annotation is inspect.Signature.empty:
        raise ValueError("Function must have return type annotation")
    
    # Collect TypeVar bindings from arguments using annotation-driven approach
    bindings: Dict[TypeVar, type] = {}
    
    # Process positional arguments
    param_names = list(sig.parameters.keys())
    for i, arg in enumerate(args):
        if i < len(param_names):
            param = sig.parameters[param_names[i]]
            if param.annotation != inspect.Parameter.empty:
                _extract_typevar_bindings_from_annotation(param.annotation, arg, bindings)
    
    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                _extract_typevar_bindings_from_annotation(param.annotation, value, bindings)
    
    # Apply type overrides (these take precedence)
    bindings.update(type_overrides)
    
    # Substitute TypeVars in return annotation
    return _substitute_type_vars(return_annotation, bindings)
