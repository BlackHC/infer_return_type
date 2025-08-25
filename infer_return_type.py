import inspect
import typing
from typing import Any, Dict, Optional, TypeVar, get_origin, get_args

from generic_utils import (
    get_generic_info, get_instance_generic_info, extract_all_typevars, 
    create_union_if_needed, is_union_type, GenericInfo,
    get_annotation_value_pairs
)


class TypeInferenceError(Exception):
    """Raised when type inference fails."""


# Use is_union_type from generic_utils instead of local function


def _extract_resolved_type(arg: Any) -> Any:
    """Extract resolved type from GenericInfo objects or return the argument as-is."""
    return arg.resolved_type if hasattr(arg, 'resolved_type') else arg


def _get_concrete_type_for_typevar_binding(value: Any) -> type:
    """Extract concrete type from a value for TypeVar binding."""
    if value is None:
        return type(None)
    
    instance_info = get_instance_generic_info(value)
    return instance_info.resolved_type or instance_info.origin


def _bind_typevar_with_conflict_check(
    typevar: TypeVar, 
    concrete_type: type, 
    bindings: Dict[TypeVar, type]
) -> None:
    """Bind a TypeVar to a concrete type, creating unions for conflicts."""
    if typevar in bindings:
        existing_type = bindings[typevar]
        if existing_type != concrete_type:
            # Create union of existing and new types instead of failing
            union_type = create_union_if_needed({existing_type, concrete_type})
            bindings[typevar] = union_type
    else:
        bindings[typevar] = concrete_type


# Removed _create_union_type - use create_union_if_needed directly


def _handle_empty_collection_typevar(
    typevar: TypeVar,
    type_overrides: Dict[TypeVar, type],
    bindings: Dict[TypeVar, type]
) -> bool:
    """
    Handle TypeVar binding for empty collections with type overrides.
    Returns True if handled (binding was made or skipped), False if should continue processing.
    """
    if typevar in type_overrides:
        concrete_type = type_overrides[typevar]
        _bind_typevar_with_conflict_check(typevar, concrete_type, bindings)
        return True
    # If no type override, skip binding - let other parameters potentially bind this TypeVar
    return True


def _infer_and_bind_collection_typevar(
    typevar: TypeVar,
    collection_items: list,
    bindings: Dict[TypeVar, type]
) -> None:
    """Infer type from collection items and bind TypeVar using generic_utils."""
    if not collection_items:
        return  # Should have been handled by _handle_empty_collection_typevar
    
    # Use generic_utils to get full type information for each item
    item_generic_infos = {get_instance_generic_info(item) for item in collection_items}
    
    # Create union from all item types
    union_info = GenericInfo.make_union_if_needed(item_generic_infos)
    concrete_type = union_info.resolved_type
    
    _bind_typevar_with_conflict_check(typevar, concrete_type, bindings)


def _handle_union_container_elements(
    union_args: list,
    container_items: list,
    bindings: Dict[TypeVar, type],
    type_overrides: Dict[TypeVar, type]
) -> None:
    """
    Handle container elements with Union type annotations by partitioning elements
    based on existing TypeVar bindings.
    """
    # If all union args are TypeVars, try to partition elements based on existing bindings
    union_typevars = [arg for arg in union_args if isinstance(arg, TypeVar)]
    
    if len(union_typevars) == len(union_args):
        # All union members are TypeVars - try to partition elements
        unbound_items = list(container_items)
        
        # For each already-bound TypeVar, filter out matching elements
        for typevar in union_typevars:
            if typevar in bindings:
                bound_type = bindings[typevar]
                # Find items that match this TypeVar's bound type
                matching_items = []
                remaining_items = []
                
                for item in unbound_items:
                    item_type = type(item)
                    if _type_matches_bound_type(item_type, bound_type):
                        matching_items.append(item)
                    else:
                        remaining_items.append(item)
                
                unbound_items = remaining_items
        
        # For remaining unbound TypeVars, try to bind them to remaining items
        unbound_typevars = [tv for tv in union_typevars if tv not in bindings]
        if unbound_typevars and unbound_items:
            if len(unbound_typevars) == 1:
                # Single unbound TypeVar gets all remaining items
                _infer_and_bind_collection_typevar(unbound_typevars[0], unbound_items, bindings)
            else:
                # Multiple unbound TypeVars - partition by type
                items_by_type = {}
                for item in unbound_items:
                    item_type = type(item)
                    if item_type not in items_by_type:
                        items_by_type[item_type] = []
                    items_by_type[item_type].append(item)
                
                # Assign types to TypeVars in order
                for i, typevar in enumerate(unbound_typevars):
                    if i < len(items_by_type):
                        type_items = list(items_by_type.values())[i]
                        _infer_and_bind_collection_typevar(typevar, type_items, bindings)
    else:
        # Mixed TypeVars and concrete types - use existing logic
        for item in container_items:
            for union_alternative in union_args:
                try:
                    temp_bindings = bindings.copy()
                    _extract_typevar_bindings_from_annotation(union_alternative, item, temp_bindings, type_overrides)
                    bindings.update(temp_bindings)
                    break  # Successfully matched this item
                except (TypeInferenceError, AttributeError, TypeError):
                    continue


def _type_matches_bound_type(item_type: type, bound_type: type) -> bool:
    """Check if an item type matches a bound type (including union types)."""
    if item_type == bound_type:
        return True
    
    # Check if bound_type is a union and item_type is one of its members
    bound_origin = get_origin(bound_type)
    
    if is_union_type(bound_origin):
        bound_args = get_args(bound_type)
        return item_type in bound_args
    
    return False


def _extract_typevar_bindings_from_annotation(
    param_annotation: Any, 
    arg_value: Any, 
    bindings: Dict[TypeVar, type],
    type_overrides: Optional[Dict[TypeVar, type]] = None
) -> None:
    """
    Extract TypeVar bindings by walking the annotation structure and binding 
    TypeVars to concrete types derived from the argument value.
    
    This is annotation-driven: we trust the annotation structure completely
    and only use the value to provide concrete type information for TypeVar binding.
    """
    
    if type_overrides is None:
        type_overrides = {}
    
    # Base case: Direct TypeVar
    if isinstance(param_annotation, TypeVar):
        # Check if we have a type override for this TypeVar
        if param_annotation in type_overrides:
            concrete_type = type_overrides[param_annotation]
        else:
            concrete_type = _get_concrete_type_for_typevar_binding(arg_value)
        
        _bind_typevar_with_conflict_check(param_annotation, concrete_type, bindings)
        return
    
    # Use generic_utils to get annotation structure
    info = get_generic_info(param_annotation)
    origin = info.origin
    
    # Handle case where args are GenericInfo objects - extract resolved types
    args = [_extract_resolved_type(arg) for arg in info.concrete_args]
    
    # Handle Union types (including Optional[T] which is Union[T, None])
    if is_union_type(origin):
        # Handle Optional[T] -> Union[T, None] as a special case
        if len(args) == 2 and type(None) in args:
            if arg_value is not None:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                _extract_typevar_bindings_from_annotation(non_none_type, arg_value, bindings, type_overrides)
            # If arg_value is None, we can't bind the TypeVar but that's OK for Optional
            return
        
        # Handle general Union types - try each alternative until one succeeds
        for union_alternative in args:
            try:
                # Create a temporary bindings copy to test this alternative
                temp_bindings = bindings.copy()
                _extract_typevar_bindings_from_annotation(union_alternative, arg_value, temp_bindings, type_overrides)
                # If we succeeded, update the real bindings and return
                bindings.update(temp_bindings)
                return
            except (TypeInferenceError, AttributeError, TypeError):
                # This alternative didn't match, try the next one
                continue
        
        # If no alternative matched, raise an error
        raise TypeInferenceError(
            f"Value {arg_value} of type {type(arg_value)} doesn't match any alternative in Union {args}"
        )
    
    # If not a Union but still not generic according to generic_utils, 
    # check if it's actually a custom generic we should handle
    if not info.is_generic:
        # Try custom generic handling as fallback
        _extract_from_custom_generic_unified(param_annotation, arg_value, bindings, type_overrides)
        return
    
    # Handle generic collections using unified logic
    if origin in (list, typing.List):
        if len(args) == 1 and isinstance(arg_value, list):
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                if not arg_value:  # Empty list
                    _handle_empty_collection_typevar(element_annotation, type_overrides, bindings)
                else:
                    _infer_and_bind_collection_typevar(element_annotation, arg_value, bindings)
            else:
                # Process each element with non-TypeVar annotation
                for item in arg_value:
                    _extract_typevar_bindings_from_annotation(element_annotation, item, bindings, type_overrides)
        else:
            raise TypeInferenceError(f"Expected list but got {type(arg_value)}")

    elif origin in (dict, typing.Dict):
        if len(args) == 2 and isinstance(arg_value, dict):
            key_annotation, value_annotation = args[0], args[1]
            
            # Handle TypeVar annotations for keys and values separately
            if isinstance(key_annotation, TypeVar):
                if not arg_value:  # Empty dict
                    _handle_empty_collection_typevar(key_annotation, type_overrides, bindings)
                else:
                    _infer_and_bind_collection_typevar(key_annotation, list(arg_value.keys()), bindings)
            else:
                # Process keys with non-TypeVar annotation
                for key in arg_value.keys():
                    _extract_typevar_bindings_from_annotation(key_annotation, key, bindings, type_overrides)
            
            if isinstance(value_annotation, TypeVar):
                if not arg_value:  # Empty dict (already checked above but being explicit)
                    _handle_empty_collection_typevar(value_annotation, type_overrides, bindings)
                else:
                    _infer_and_bind_collection_typevar(value_annotation, list(arg_value.values()), bindings)
            else:
                # Process values with non-TypeVar annotation
                for value in arg_value.values():
                    _extract_typevar_bindings_from_annotation(value_annotation, value, bindings, type_overrides)
        else:
            raise TypeInferenceError(f"Expected dict but got {type(arg_value)}")

    elif origin in (tuple, typing.Tuple):
        if isinstance(arg_value, tuple):
            # Handle tuple[T, ...] vs tuple[T1, T2, T3]
            if len(args) == 2 and args[1] is ...:
                # Variable length tuple: tuple[T, ...]
                element_annotation = args[0]
                
                if isinstance(element_annotation, TypeVar):
                    if not arg_value:  # Empty tuple
                        _handle_empty_collection_typevar(element_annotation, type_overrides, bindings)
                    else:
                        _infer_and_bind_collection_typevar(element_annotation, list(arg_value), bindings)
                else:
                    # Process each element with non-TypeVar annotation
                    for item in arg_value:
                        _extract_typevar_bindings_from_annotation(element_annotation, item, bindings, type_overrides)
            else:
                # Fixed length tuple: tuple[T1, T2, T3]
                for i, item in enumerate(arg_value):
                    if i < len(args):
                        _extract_typevar_bindings_from_annotation(args[i], item, bindings, type_overrides)
        else:
            raise TypeInferenceError(f"Expected tuple but got {type(arg_value)}")

    elif origin in (set, typing.Set):
        if len(args) == 1 and isinstance(arg_value, set):
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                if not arg_value:  # Empty set
                    _handle_empty_collection_typevar(element_annotation, type_overrides, bindings)
                else:
                    _infer_and_bind_collection_typevar(element_annotation, list(arg_value), bindings)
            else:
                # Check if element annotation is a Union type
                elem_info = get_generic_info(element_annotation)
                if is_union_type(elem_info.origin):
                    # Handle Union element types with smart partitioning
                    union_args = [_extract_resolved_type(arg) for arg in elem_info.concrete_args]
                    _handle_union_container_elements(union_args, list(arg_value), bindings, type_overrides)
                else:
                    # Process each element with non-Union annotation
                    for item in arg_value:
                        _extract_typevar_bindings_from_annotation(element_annotation, item, bindings, type_overrides)
        else:
            raise TypeInferenceError(f"Expected set but got {type(arg_value)}")
    
    # Handle custom generic types using generic_utils
    else:
        _extract_from_custom_generic_unified(param_annotation, arg_value, bindings, type_overrides)


def _extract_from_custom_generic_unified(
    param_annotation: Any, 
    arg_value: Any, 
    bindings: Dict[TypeVar, type],
    type_overrides: Optional[Dict[TypeVar, type]] = None  # pylint: disable=unused-argument
) -> None:
    """
    Extract TypeVar bindings from custom generic types using unified generic_utils approach.
    
    This replaces the old special-case logic with a unified approach.
    """
    
    # Get annotation info
    annotation_info = get_generic_info(param_annotation)
    
    # Get instance info  
    instance_info = get_instance_generic_info(arg_value)
    
    # Try to align annotation concrete_args with instance concrete_args positionally
    if annotation_info.concrete_args and instance_info.concrete_args:
        if len(annotation_info.concrete_args) == len(instance_info.concrete_args):
            for ann_arg, inst_arg in zip(annotation_info.concrete_args, instance_info.concrete_args):
                # For nested alignment, work with GenericInfo objects directly
                _align_nested_structures(ann_arg, inst_arg, bindings)
            return

    # If no concrete args in instance but annotation has TypeVars, try to extract from field values
    if annotation_info.type_params and not instance_info.concrete_args:
        _extract_from_instance_fields(param_annotation, arg_value, bindings)
        return
    
    # Fallback: try to extract TypeVars from the annotation structure and bind them to 
    # inferred types from the instance
    all_typevars = extract_all_typevars(param_annotation)
    if len(all_typevars) == 1:
        # Simple case: single TypeVar, try to infer from instance
        typevar = all_typevars[0]
        concrete_type = _get_concrete_type_for_typevar_binding(arg_value)
        _bind_typevar_with_conflict_check(typevar, concrete_type, bindings)


def _extract_from_instance_fields(annotation: Any, value: Any, bindings: Dict[TypeVar, type]) -> None:
    """
    Extract TypeVar bindings from instance field/element values using the unified annotation-value pairs interface.
    
    This function handles cases where a generic instance doesn't have explicit type parameters
    but we can infer the types from the field/element values.
    """
    # Get (GenericInfo, value) pairs that map annotation type parameters to concrete values
    annotation_value_pairs = get_annotation_value_pairs(annotation, value)
    
    if not annotation_value_pairs:
        return  # No mappings found
    
    # Group values by TypeVar (GenericInfo with TypeVar origin)
    typevar_to_values = {}
    
    for generic_info, concrete_value in annotation_value_pairs:
        if isinstance(generic_info.origin, TypeVar):
            typevar = generic_info.origin
            if typevar not in typevar_to_values:
                typevar_to_values[typevar] = []
            typevar_to_values[typevar].append(concrete_value)
    
    # Bind each TypeVar to the inferred type from its values
    for typevar, values in typevar_to_values.items():
        if values:
            # Get types from all values and create union if needed
            value_types = {type(v) for v in values if v is not None}
            if value_types:
                inferred_type = create_union_if_needed(value_types)
                _bind_typevar_with_conflict_check(typevar, inferred_type, bindings)


def _align_nested_structures(annotation_info: Any, concrete_info: Any, bindings: Dict[TypeVar, type]) -> None:
    """
    Recursively align annotation structure with concrete type structure to extract TypeVar bindings.
    
    Works with both GenericInfo objects and raw types.
    """
    
    # Handle GenericInfo objects
    if hasattr(annotation_info, 'origin'):
        ann_origin = annotation_info.origin
        ann_args = annotation_info.concrete_args
    else:
        ann_origin = annotation_info
        ann_args = []
        
    if hasattr(concrete_info, 'origin'):
        concrete_origin = concrete_info.origin
        concrete_args = concrete_info.concrete_args
    else:
        concrete_origin = concrete_info
        concrete_args = []
    
    # Direct TypeVar case
    if isinstance(ann_origin, TypeVar):
        concrete_type = concrete_info.resolved_type if hasattr(concrete_info, 'resolved_type') else concrete_origin
        _bind_typevar_with_conflict_check(ann_origin, concrete_type, bindings)
        return
    
    # If origins don't match, we can't align
    if ann_origin != concrete_origin:
        return
    
    # If different number of args, we can't align
    if len(ann_args) != len(concrete_args):
        return
    
    # Recursively align each argument
    for ann_arg, concrete_arg in zip(ann_args, concrete_args):
        _align_nested_structures(ann_arg, concrete_arg, bindings)


def _substitute_type_vars(annotation: Any, bindings: Dict[TypeVar, type]) -> Any:
    """Substitute TypeVars in an annotation with their concrete bindings."""
    
    if isinstance(annotation, TypeVar):
        if annotation in bindings:
            return bindings[annotation]
        else:
            raise TypeInferenceError(f"Unbound TypeVar: {annotation}")
    
    # Use generic_utils to handle the structure
    info = get_generic_info(annotation)
    
    if not info.is_generic:
        return annotation
    
    origin = info.origin
    args = info.concrete_args
    
    # Handle Union types specially
    if is_union_type(origin):
        substituted_args = []
        
        for arg in args:
            try:
                resolved_arg = _extract_resolved_type(arg)
                substituted_arg = _substitute_type_vars(resolved_arg, bindings)
                substituted_args.append(substituted_arg)
            except TypeInferenceError:
                # Skip unbound TypeVars in unions
                continue
        
        # If we have at least one bound arg, return the union of bound args
        if substituted_args:
            if len(substituted_args) == 1:
                return substituted_args[0]
            # Create Union type directly
            return create_union_if_needed(set(substituted_args))
        
        # If no args were bound, this is an error
        raise TypeInferenceError(f"No TypeVars bound in Union {args}")
    
    # Recursively substitute in type arguments
    substituted_args = []
    for arg in args:
        resolved_arg = _extract_resolved_type(arg)
        substituted_args.append(_substitute_type_vars(resolved_arg, bindings))
    
    # Use generic_utils to reconstruct the type properly
    try:
        # Create a new GenericInfo with substituted args
        substituted_generic_args = [
            GenericInfo(origin=arg) if not isinstance(arg, GenericInfo) else arg
            for arg in substituted_args
        ]
        new_info = GenericInfo(origin=origin, concrete_args=substituted_generic_args)
        return new_info.resolved_type
    except (TypeError, ValueError, AttributeError):
        # Fallback to manual reconstruction for built-in types
        if origin in (list, typing.List):
            return list[substituted_args[0]]
        elif origin in (dict, typing.Dict):
            return dict[substituted_args[0], substituted_args[1]]
        elif origin in (tuple, typing.Tuple):
            return tuple[tuple(substituted_args)]
        elif origin in (set, typing.Set):
            return set[substituted_args[0]]
        else:
            # For other generic types, try direct reconstruction
            try:
                return origin[tuple(substituted_args)]
            except (TypeError, ValueError, AttributeError):
                # If reconstruction fails, return the original annotation
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
                _extract_typevar_bindings_from_annotation(param.annotation, arg, bindings, type_overrides)
    
    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                _extract_typevar_bindings_from_annotation(param.annotation, value, bindings, type_overrides)
    
    # Apply type overrides (these take precedence)
    bindings.update(type_overrides)
    
    # Substitute TypeVars in return annotation
    return _substitute_type_vars(return_annotation, bindings)
