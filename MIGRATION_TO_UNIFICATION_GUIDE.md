# Migration to Unification Implementation Guide

## Executive Summary

The unification implementation is **ready for migration** with 3 critical fixes needed. This guide provides a clear path to making it the primary implementation.

### Current Status
- ✅ **40 passing tests** - solid foundation
- ⚠️ **3 critical bugs** - must fix before production
- 📝 **7 missing features** - should port from CSP
- 🔬 **9 investigation items** - nice to have

## Critical Issues (Must Fix)

### 1. Conflicting TypeVar Bindings → Union Formation
**File**: `unification_type_inference.py`, line ~788

**Current Behavior**:
```python
def identity(a: A, b: A) -> A: ...
infer_return_type_unified(identity, 1, 'x')
# Raises: TypeInferenceError - Conflicting type assignments
```

**Expected Behavior**:
```python
# Should return: int | str
```

**Root Cause**: In `_resolve_typevar_constraints()`, when multiple INVARIANT constraints have different types, it raises an error instead of forming a union.

**Fix Location**:
```python
# Around line 796-799
if len(invariant_constraints) > 1:
    invariant_types = [c.concrete_type for c in invariant_constraints]
    if len(set(invariant_types)) > 1:
        # Current: raises UnificationError
        # Should: return create_union_if_needed(set(invariant_types))
```

**Test to Unskip**: `test_conflicting_typevar_should_create_union`

---

### 2. None Filtering in Optional[A]
**File**: `unification_type_inference.py`, line ~772-780

**Current Behavior**:
```python
def process(d: Dict[str, Optional[A]]) -> A: ...
infer_return_type_unified({'a': 1, 'b': None, 'c': 2})
# Returns: int | None (wrong!)
```

**Expected Behavior**:
```python
# Should return: int (filter out None)
```

**Root Cause**: When processing constraints, None is included in the union instead of being filtered out.

**Fix Location**:
```python
# Around line 773-780
none_types = [c for c in constraints if c.concrete_type == type(None)]
non_none_constraints = [c for c in constraints if c.concrete_type != type(None)]

if none_types and non_none_constraints:
    # Current: includes None in union
    # Should: only use non_none_types for A, None is handled by Optional wrapper
    non_none_types = [c.concrete_type for c in non_none_constraints]
    return self._check_typevar_bounds(typevar, create_union_if_needed(set(non_none_types)))
```

**Test to Unskip**: `test_none_filtering_in_optional`

---

### 3. Complex Union Structures
**File**: `unification_type_inference.py`, `_handle_union_value_binding`

**Current Behavior**:
```python
def extract(data: Dict[str, Union[A, List[A], Dict[str, A]]]) -> A: ...
test_data = {'single': 42, 'list': [43, 44], 'nested': {'value': 45}}
infer_return_type_unified(extract, test_data)
# Raises: TypeInferenceError - Conflicting type assignments
```

**Expected Behavior**:
```python
# Should return: int (A is consistently int across all positions)
```

**Root Cause**: When A appears in multiple positions within a union (`A`, `List[A]`, `Dict[str, A]`), the system creates multiple INVARIANT constraints that conflict.

**Fix Approach**: 
1. Recognize when constraints come from the same logical source (same TypeVar in different positions of a union)
2. These should be treated as COVARIANT constraints that can form unions
3. Or better: extract the "inner" TypeVar uniformly before creating constraints

**Test to Unskip**: `test_complex_union_structure`

---

## Priority Feature Ports from CSP (Should Do)

### 4. Explicit Variance Support
**Effort**: 2-3 days
**Tests to Port**: 
- `test_covariant_variance_explicit`
- `test_contravariant_variance_explicit`
- `test_invariant_dict_keys`

**What to Port**:
- Add explicit Variance enum (COVARIANT, CONTRAVARIANT, INVARIANT)
- Make constraint resolution variance-aware
- Add variance rules for common types (List=covariant, Dict keys=invariant, etc.)

**Benefits**:
- Better handling of subtype relationships
- More precise type inference
- Catches more type errors

### 5. Constraint Priority System
**Effort**: 1 day
**Test to Port**: `test_constraint_priority_resolution`

**What to Port**:
- Add priority field to Constraint class
- Make type_overrides highest priority
- Resolve constraints in priority order

**Benefits**:
- type_overrides work more reliably
- Predictable behavior with complex constraints

### 6. Domain-Based Reasoning
**Effort**: 2 days
**Test to Port**: `test_domain_filtering_with_constraints`

**What to Port**:
- TypeDomain class for tracking possible types
- Filtering based on bounds/constraints
- More sophisticated constraint propagation

**Benefits**:
- Better handling of TypeVar bounds
- More efficient constraint solving

### 7. Union Distribution Logic
**Effort**: 1-2 days
**Tests to Port**:
- `test_union_or_logic_distribution`
- `test_subset_constraints`

**What to Port**:
- OR logic for Union type alternatives
- Subset constraints (`{A, B} ⊇ {int, str}`)
- Smart type distribution among TypeVars

**Benefits**:
- Better `Set[Union[A, B]]` handling
- More flexible union type inference

---

## Implementation Plan

### Phase 1: Critical Fixes (Week 1)
**Goal**: Fix the 3 blocking issues

**Day 1-2**: Fix conflicting TypeVar bindings
- Modify `_resolve_typevar_constraints()` to create unions
- Update tests and verify no regressions
- Unskip `test_conflicting_typevar_should_create_union`

**Day 3**: Fix None filtering
- Update constraint resolution to filter None in Optional contexts
- Unskip `test_none_filtering_in_optional`

**Day 4-5**: Fix complex union structures
- Refactor `_handle_union_value_binding()` 
- Add logic to recognize same-typevar-different-positions
- Unskip `test_complex_union_structure`

**Deliverable**: All critical tests passing, 43 of 59 tests passing

### Phase 2: Feature Parity (Week 2-3)
**Goal**: Port key features from CSP

**Days 6-8**: Add variance support
- Port Variance enum and rules
- Update constraint resolution
- Unskip variance tests

**Days 9-10**: Add constraint priorities
- Add priority system
- Update constraint solver
- Unskip priority test

**Days 11-13**: Add domain reasoning and union distribution
- Port TypeDomain logic
- Add union distribution
- Unskip remaining CSP feature tests

**Deliverable**: 50+ of 59 tests passing, feature parity with CSP

### Phase 3: Polish (Week 4)
**Goal**: Production ready with good DX

**Days 14-15**: Improve error messages
- Add constraint traces to errors
- Make messages human-readable
- Unskip error message tests

**Days 16-17**: Add performance tests and optimization
- Run benchmarks
- Optimize if needed
- Unskip performance tests

**Days 18-20**: Documentation and migration guide
- Update all docs
- Write migration guide for users
- Deprecation plan for old implementations

**Deliverable**: Production-ready, all 59 tests passing

---

## Testing Strategy

### Regression Testing
After each fix:
1. Run all 3 implementation comparison tests
2. Verify unification passes same tests as original/CSP
3. Check for performance regressions

### Integration Testing
- Test with real-world codebases
- Test with complex generic libraries (Pydantic, dataclasses, etc.)
- Test edge cases from user feedback

### Performance Testing
- Benchmark against original and CSP
- Test with deeply nested types
- Test with many TypeVars

---

## Migration Path for Users

### Version 1.0 (After Phase 1)
- Unification available as `infer_return_type_unified`
- Original still default
- Document known limitations

### Version 1.5 (After Phase 2)
- Unification becomes default
- Original available as `infer_return_type_legacy`
- Migration guide for breaking changes

### Version 2.0 (After Phase 3)
- Unification is only implementation
- Remove legacy code
- Clean public API

---

## Risk Mitigation

### Breaking Changes
**Risk**: Some edge cases might behave differently
**Mitigation**: 
- Comprehensive comparison testing
- Clear migration guide
- Deprecation period with both implementations available

### Performance Regression
**Risk**: Unification might be slower
**Mitigation**:
- Performance benchmarks before migration
- Optimize hot paths
- Add caching if needed

### Unforeseen Edge Cases
**Risk**: Real-world code might expose new bugs
**Mitigation**:
- Beta period with opt-in unification
- Collect feedback
- Fix issues before making it default

---

## Success Criteria

### Must Have (Phase 1)
- ✅ All 3 critical tests passing
- ✅ No regressions in passing tests
- ✅ Comparison tests show parity with original/CSP

### Should Have (Phase 2)
- ✅ Variance support
- ✅ Constraint priorities
- ✅ 50+ of 59 tests passing

### Nice to Have (Phase 3)
- ✅ Great error messages
- ✅ Performance competitive with original
- ✅ All 59 tests passing
- ✅ Comprehensive documentation

---

## Conclusion

The unification implementation is **close to production ready**. With focused effort on the 3 critical issues, it can become the primary implementation within 1-2 weeks. The additional CSP features can be ported over 2-3 weeks for full feature parity.

**Recommended Action**: Start Phase 1 immediately. The fixes are well-understood and localized.
