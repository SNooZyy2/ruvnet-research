/**
 * FTS5 Query Sanitizer — extracted from OCR-Provenance (bm25.ts:869-928)
 *
 * Prevents FTS5 injection by:
 * - Stripping all metacharacters ('"()*:^~+{}[]\\;@<>#!$%&|,./`?)
 * - Preserving AND/OR/NOT operators
 * - Treating hyphens as word separators (matches FTS5 unicode61 tokenizer)
 * - Inserting implicit AND between consecutive terms
 * - Stripping leading/trailing/consecutive operators
 * - Stripping leading NOT (prevents negative-only queries)
 *
 * Use during L3-01 (Search Context) implementation.
 * See ADR-v4-008 Rejected Alternatives section for full context.
 */
export function sanitizeFTS5Query(query: string): string {
  const FTS5_OPERATORS = new Set(['AND', 'OR', 'NOT']);
  const rawTokens = query
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0);

  const result: string[] = [];
  for (const raw of rawTokens) {
    if (FTS5_OPERATORS.has(raw.toUpperCase())) {
      result.push(raw.toUpperCase());
    } else {
      // Treat hyphens as word separators (matching FTS5 unicode61 tokenizer)
      const parts = raw
        .split(/-/)
        .map((p) => p.replace(/['"()*:^~+{}[\]\\;@<>#!$%&|,./`?]/g, ''))
        .filter((p) => p.length > 0);
      result.push(...parts);
    }
  }

  // Strip leading/trailing operators and consecutive operators
  while (result.length > 0 && FTS5_OPERATORS.has(result[0])) result.shift();
  while (result.length > 0 && FTS5_OPERATORS.has(result[result.length - 1])) result.pop();
  const cleaned: string[] = [];
  for (const t of result) {
    if (
      FTS5_OPERATORS.has(t) &&
      cleaned.length > 0 &&
      FTS5_OPERATORS.has(cleaned[cleaned.length - 1])
    )
      continue;
    cleaned.push(t);
  }

  // Strip leading NOT to prevent accidental negative-only queries
  if (cleaned.length >= 2 && cleaned[0] === 'NOT') {
    cleaned.shift();
  }

  const finalTokens = cleaned.filter((t) => t.length > 0);
  if (finalTokens.length === 0) {
    throw new Error('Query contains no valid search tokens after sanitization');
  }

  // Insert implicit AND between consecutive non-operator tokens
  const parts: string[] = [];
  for (let i = 0; i < finalTokens.length; i++) {
    parts.push(finalTokens[i]);
    if (
      i < finalTokens.length - 1 &&
      !FTS5_OPERATORS.has(finalTokens[i]) &&
      !FTS5_OPERATORS.has(finalTokens[i + 1])
    ) {
      parts.push('AND');
    }
  }

  return parts.join(' ');
}
