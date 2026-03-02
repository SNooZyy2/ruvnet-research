// scripts/validate-enums.js — ADR-v4-009 inline enum validation helper
// Zero dependencies. Copy-paste into node -e scripts, or require() from script files.
//
// Usage in agent inline scripts:
//   assertEnum('severity', 'CRITICAL');    // passes
//   assertEnum('category', 'facade');      // throws: must be FACADE
//   assertEnum('relationship', 'imports'); // throws: must be IMPORTS

const VALID = {
  relationship: new Set(['IMPORTS','USES','EXPORTS','DECLARES','SIBLINGS','COMPETES','WRAPS','FEEDS','TESTS','BROKEN']),
  category: new Set(['ARCHITECTURE','QUALITY','INTEGRATION','PERFORMANCE','ALGORITHM','FACADE','SECURITY','BUG','GENUINE','TESTING','DOCUMENTATION','INCOMPLETE']),
  severity: new Set(['CRITICAL','HIGH','MEDIUM','INFO']),
};

function assertEnum(field, value) {
  if (!VALID[field]) throw new Error(`Unknown enum field: "${field}". Must be one of: ${Object.keys(VALID).join(', ')}`);
  if (!VALID[field].has(value)) {
    throw new Error(`Invalid ${field}: "${value}". Must be one of: ${[...VALID[field]].join(', ')}`);
  }
}

module.exports = { VALID, assertEnum };
