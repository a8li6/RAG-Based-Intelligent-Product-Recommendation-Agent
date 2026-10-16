# Test fixtures

Tiny, committed, fully synthetic inputs. Keep them small; rebuild artifacts from the catalog
rather than committing them.

| File | Purpose |
|---|---|
| `invalid_catalog.jsonl` | One valid record (line 1) plus one distinct error on each of lines 2-7: negative price, unknown field, unsupported currency, duplicate ID, malformed JSON, naive timestamp. |
| `hostile_product.jsonl` | A valid product whose description contains prompt-injection text. It must never reach a user-visible fact or break a prompt section. |

The main reference catalog lives in `data/sample/products.jsonl`.
