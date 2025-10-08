# Cleanup Plan: Migration to Unification Only

## Files to Keep
- ✅ `unification_type_inference.py` - Primary implementation
- ✅ `generic_utils.py` - Utilities used by unification
- ✅ `test_infer_return_type_unified.py` - Primary test suite (60 tests)
- ✅ `test_generic_utils.py` - Tests for generic_utils
- ✅ `pyproject.toml` - Project configuration
- ✅ `README.md` - Documentation
- ✅ `uv.lock` - Lock file

## Files to Delete

### Implementation Files (Old)
- ❌ `infer_return_type.py` - Original implementation (replaced by unification)
- ❌ `csp_type_inference.py` - CSP implementation (replaced by unification)

### Test Files (Old/Redundant)
- ❌ `test_infer_return_type.py` - Tests original (migrated to unified)
- ❌ `test_infer_return_type_csp.py` - Tests CSP (migrated to unified)
- ❌ `test_csp_inference.py` - Tests CSP features (migrated to unified)
- ❌ `test_csp_improvements.py` - Tests CSP improvements (covered in unified)
- ❌ `test_unification_improvements.py` - Tests unification (migrated to unified)
- ❌ `test_comparison.py` - Simple comparison script (no longer needed)
- ❌ `test_simplified_csp.py` - CSP demo script (no longer needed)
- ❌ `test_unified_basic.py` - Basic unification demo (no longer needed)
- ❌ `test_type_erasure_spike.py` - Exploratory tests (can keep for reference or delete)
- ❌ `test_all_implementations_comparison.py` - Comparison suite (no longer needed after cleanup)

### Documentation Files (Analysis/Comparison)
- ❌ `CSP_TYPE_INFERENCE_ANALYSIS.md` - CSP analysis (superseded)
- ❌ `CSP_IMPROVEMENTS_SUMMARY.md` - CSP improvements (superseded)
- ❌ `INFER_RETURN_TYPE_IMPROVEMENTS.md` - Original improvements (superseded)
- ❌ `FINDINGS_SUMMARY.md` - Old findings (superseded)
- ❌ `UNIFICATION_DESIGN.md` - Design doc (may want to keep for history)

### Keep for Reference
- ⚠️ `IMPLEMENTATION_COMPARISON_SUMMARY.md` - Good historical reference
- ⚠️ `UNIFICATION_GAPS_ANALYSIS.md` - Useful for tracking fixes
- ⚠️ `UNIFICATION_TEST_SUMMARY.md` - Test tracking
- ⚠️ `MIGRATION_TO_UNIFICATION_GUIDE.md` - Migration guide
- ⚠️ `demo_key_differences.py` - Historical demo (can delete or keep)

## Final File Structure

```
infer_return_type/
├── unification_type_inference.py     # Primary implementation
├── generic_utils.py                   # Utilities
├── test_infer_return_type_unified.py # Primary tests (60 tests)
├── test_generic_utils.py              # Utility tests
├── pyproject.toml                     # Config
├── README.md                          # Main docs
├── uv.lock                            # Lock file
└── docs/ (optional)
    ├── IMPLEMENTATION_COMPARISON_SUMMARY.md
    ├── UNIFICATION_GAPS_ANALYSIS.md
    ├── UNIFICATION_TEST_SUMMARY.md
    └── MIGRATION_TO_UNIFICATION_GUIDE.md
```

## Cleanup Steps

1. ✅ Verify all tests ported to test_infer_return_type_unified.py
2. ⏳ Delete old implementation files
3. ⏳ Delete old test files
4. ⏳ Delete superseded documentation
5. ⏳ Update README.md with new structure
6. ⏳ Run final test suite to verify everything works

## Test Count Verification

**Before Cleanup**: Multiple test files with ~300+ total tests
**After Cleanup**: Single test file with 60 comprehensive tests
- 40 passing tests
- 19 skipped tests (documented weaknesses/TODOs)
- 1 architectural test

## Safety Checks

- ✅ All unique test scenarios captured in unified test file
- ✅ Architectural test ported
- ✅ Generic utils tests separate and maintained
- ✅ Documentation files preserved for reference
